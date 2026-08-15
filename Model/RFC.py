import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from ModelUtils.model_utils import Conv_MLP, \
    AdaLayerNorm, Transpose, GELU2, series_decomp

from ModelUtils.utils import LearnablePositionalEncoding
from torch.jit import Final
from typing import Type
from timm.layers import use_fused_attn

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TrendBlock(nn.Module):
    """
    Model trend of time series using the polynomial regressor.
    """

    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super(TrendBlock, self).__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            act,
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
        )

        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input):
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals

def topk_mask(input, k, dim, return_mask=False):
    topk, indices = torch.topk(input, k, dim=dim)
    fill = 1 if return_mask else topk
    masked = torch.zeros_like(input, device="cuda:0").scatter_(dim, indices, fill)

    return masked

class NeuralFourierLayer(nn.Module):
    def __init__(self, in_dim, out_dim, seq_len=168, pred_len=24):
        super().__init__()

        self.out_len = seq_len
        self.freq_num = (seq_len // 2) + 1

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight = nn.Parameter(torch.empty((self.freq_num, in_dim, out_dim), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.freq_num, out_dim), dtype=torch.cfloat))
        self.init_parameters()

    def forward(self, x_emb, mask=True):
        x_fft = torch.fft.rfft(x_emb, dim=1)[:, :self.freq_num]
        output_fft = x_fft
        if mask:
            amp = output_fft.abs().permute(0, 2, 1).reshape((-1, self.freq_num))
            output_fft_mask = topk_mask(amp, k=8, dim=1, return_mask=True)
            tmp_outdim = 64
            output_fft_mask = output_fft_mask.reshape(x_emb.shape[0], tmp_outdim, self.freq_num).permute(0, 2, 1)
            output_fft = output_fft * output_fft_mask
        return torch.fft.irfft(output_fft, n=self.out_len, dim=1)

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)



class EncoderBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU'
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.att = Attention(n_embd, num_heads=n_head, qkv_bias=True)

        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, timestep, mask=None, label_emb=None):
        a = self.att(self.ln1(x, timestep, label_emb))
        x = x + a
        x = x + self.mlp(self.ln2(x))  # only one really use encoder_output
        return x


class Encoder(nn.Module):
    def __init__(
            self,
            n_layer=14,
            n_embd=1024,
            n_head=16,
            attn_pdrop=0.,
            resid_pdrop=0.,
            mlp_hidden_times=4,
            block_activate='GELU',
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[EncoderBlock(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times,
            activate=block_activate,
        ) for _ in range(n_layer)])

    def forward(self, input, t, padding_masks=None, label_emb=None):
        x = input
        for block_idx in range(len(self.blocks)):
            x = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x


class DecoderBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self,
                 n_channel,
                 n_feat,
                 n_embd=1024,
                 n_head=16,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.att = Attention(n_embd, num_heads=n_head, qkv_bias=True)

        self.ln1_1 = AdaLayerNorm(n_embd)

        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_feat, act=act)
        self.seasonal = NeuralFourierLayer(n_embd, n_channel, n_channel, n_channel)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.proj = nn.Conv1d(n_channel, n_channel * 2, 1)
        self.conseq_fortrend = nn.Sequential(
            Transpose(shape=(1, 2)),
            nn.Linear(n_channel, mlp_hidden_times * n_channel),
            act,
            nn.Linear(mlp_hidden_times * n_channel, n_channel),
            nn.Dropout(resid_pdrop),
            Transpose(shape=(1, 2)),
            nn.Conv1d(n_channel, n_channel, 1),
        )
        self.conseq_forseason = nn.Sequential(
            Transpose(shape=(1, 2)),
            nn.Linear(n_channel, mlp_hidden_times * n_channel),
            act,
            nn.Linear(mlp_hidden_times * n_channel, n_channel),
            nn.Dropout(resid_pdrop),
            Transpose(shape=(1, 2)),
            nn.Conv1d(n_channel, n_channel, 1),
        )

        self.linear = nn.Linear(n_embd, n_feat)

    def forward(self, encoder_output, timestep, mask=None, label_emb=None):
        att_input = self.ln1(encoder_output, timestep)
        att = self.att(att_input)
        x = encoder_output + att


        trendx = self.conseq_fortrend(x)
        trend = self.trend(trendx)
        seasonx = self.conseq_forseason(x)
        season = self.seasonal(seasonx)

        x = x + self.mlp(self.ln2(x))
        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m), trend, season


class TrSe_Decoder(nn.Module):
    def __init__(
            self,
            n_channel,
            n_feat,
            n_embd=1024,
            n_head=16,
            n_layer=10,
            resid_pdrop=0.1,
            mlp_hidden_times=4,
            block_activate='GELU',
    ):
        super().__init__()
        self.d_model = n_embd
        self.n_feat = n_feat
        self.blocks = nn.Sequential(*[DecoderBlock(
            n_feat=n_feat,
            n_channel=n_channel,
            n_embd=n_embd,
            n_head=n_head,
            resid_pdrop=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times,
            activate=block_activate,
        ) for _ in range(n_layer)])

        self.conv_ajback = Conv1d_Ajust(n_embd, n_feat, resid_pdrop=resid_pdrop)
        self.season_res = nn.Conv1d(n_embd, n_feat, kernel_size=1, stride=1, padding=0,padding_mode='circular', bias=False)
        self.trend_res = nn.Conv1d(n_layer, 1, kernel_size=1, stride=1, padding=0,padding_mode='circular', bias=False)

    def forward(self, enc, t, padding_masks=None, label_emb=None):
        b, c, _ = enc.shape
        mean = []
        season = torch.zeros((b, c, self.d_model), device=enc.device)
        trend = torch.zeros((b, c, self.n_feat), device=enc.device)
        for block_idx in range(len(self.blocks)):
            x, residual_mean, residual_trend, residual_season = \
                self.blocks[block_idx](enc, t, mask=padding_masks, label_emb=label_emb)
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)

        mean = torch.cat(mean, dim=1)

        x_ori = self.conv_ajback(x)
        x_mean = torch.mean(x_ori, dim=1, keepdim=True)

        season = self.season_res(season.transpose(1, 2)).transpose(1, 2) + x_ori - x_mean
        trend = self.trend_res(mean) + x_mean + trend

        return trend, season


class Conv1d_Ajust(nn.Module):
    def __init__(self, in_dim, out_dim, resid_pdrop=0.):
        super().__init__()
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_dim, out_dim, 3, stride=1, padding=1),
            nn.Dropout(p=resid_pdrop),
        )

    def forward(self, x):
        return self.sequential(x).transpose(1, 2)

class RFC(nn.Module):
    def __init__(
            self,
            n_feat,
            n_channel,
            n_layer_enc=5,
            n_layer_dec=14,
            n_embd=1024,
            n_heads=16,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
            mlp_hidden_times=4,
            block_activate='GELU',
            max_len=2048,
            **kwargs
    ):
        super().__init__()
        self.embedding = nn.Sequential(
            Conv1d_Ajust(n_feat, n_embd, resid_pdrop=resid_pdrop),
            LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)
        )

        self.encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate)

        self.decoder = TrSe_Decoder(n_channel, n_feat, n_embd, n_heads, n_layer_dec, resid_pdrop,mlp_hidden_times,block_activate)



    def forward(self, input, t, padding_masks=None, return_res=False):
        inp_enc = self.embedding(input)
        enc_cond = self.encoder(inp_enc, t, padding_masks=padding_masks)

        trend, season = self.decoder(enc_cond, t, padding_masks=padding_masks)
        return trend, season




if __name__ == '__main__':
    pass