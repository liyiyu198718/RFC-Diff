import math
import torch

from torch import nn
from tqdm.auto import tqdm
from functools import partial

from Model.RFC import RFC
from ModelUtils.model_utils import default, identity, extract
import numpy as np

import csv


import torch.nn.functional as F



def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class RFC_Diff(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            timesteps=1000,
            sampling_timesteps=None,
            beta_schedule='cosine',
            n_heads=4,
            mlp_hidden_times=4,
            eta=0.,
            attn_pd=0.,
            resid_pd=0.,
            use_ff=True,
            reg_weight=None,
            aug_times=1,
            kernel_size=3,
            **kwargs
    ):
        super(RFC_Diff, self).__init__()

        self.aug_times = aug_times
        self.kernel_size = kernel_size

        self.eta, self.use_ff = eta, use_ff
        self.seq_length = seq_length
        self.feature_size = feature_size

        self.model = RFC(n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc,
                                 n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd,
                                 mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, n_embd=d_model,  **kwargs)



        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)



        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate reweighting

        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)

        self.counttrain = 0
        self.counttest = 0



        self.testcount_ch = {}
        self.testcount_norm = {}

        self.greater1_total = 0
        self.less_neg1_total = 0
        self.greater1_total2 = 0
        self.less_neg1_total2 = 0

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def output(self, x, t, padding_masks=None):
        trend, season = self.model(x, t, padding_masks=padding_masks)
        model_output = trend + season
        # season = self.model(x, t, padding_masks=padding_masks)
        # model_output = season
        return model_output



    def model_predictions(self, x, t, clip_x_start=False, padding_masks=None):
        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start = self.output(x, t, padding_masks)

        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True):
        _, x_start = self.model_predictions(x, t)
        if clip_denoised:
            x_start.clamp_(-1., 1.)


        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(self, x, t: int, clip_denoised=True, cond_fn=None, model_kwargs=None):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        if cond_fn is not None:
            model_mean = self.condition_mean(
                cond_fn, model_mean, model_log_variance, x, t=batched_times, model_kwargs=model_kwargs
            )
        pred_series = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_series, x_start

    @torch.no_grad()
    def sample(self, shape):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, _ = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img

    def generate_mts(self, batch_size=16, model_kwargs=None, cond_fn=None):
        feature_size, seq_length = self.feature_size, self.seq_length
        if cond_fn is not None:
            sample_fn = self.fast_sample_cond if self.fast_sampling else self.sample_cond
            return sample_fn((batch_size, seq_length, feature_size), model_kwargs=model_kwargs, cond_fn=cond_fn)
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        return sample_fn((batch_size, seq_length, feature_size))



    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def test_result(self):
        if self.testcount_norm.get("aver_corre") is not None:

            aver_corre_norm_str = "OpenNorm aver_corre= %.3f" %(self.testcount_norm["aver_corre"])
            print(aver_corre_norm_str)
            aver_corre_mini_norm_str = "OpenNorm aver_corre_mini= %.3f" %(self.testcount_norm["aver_corre_mini"])
            print(aver_corre_mini_norm_str)
            lower_0_norm_str = "OpenNorm lower_0= %.3f" %(self.testcount_norm["lower_0"])
            print(lower_0_norm_str)
            lower_20_norm_str = "OpenNorm lower_20= %.3f" %(self.testcount_norm["lower_20"])
            print(lower_20_norm_str)
            lower_40_norm_str = "OpenNorm lower_40= %.3f" %(self.testcount_norm["lower_40"])
            print(lower_40_norm_str)
            lower_60_norm_str = "OpenNorm lower_60= %.3f" %(self.testcount_norm["lower_60"])
            print(lower_60_norm_str)
            lower_80_norm_str = "OpenNorm lower_80= %.3f" %(self.testcount_norm["lower_80"])
            print(lower_80_norm_str)
            lower_100_norm_str = "OpenNorm lower_100= %.3f" %(self.testcount_norm["lower_100"])
            print(lower_100_norm_str)

        self.testcount_norm = {}





    def interpolate_3d_tensor(self,x_start_ori):


        x_transposed = x_start_ori.transpose(1, 2)



        beishu = int(self.seq_length / len(x_start_ori[0]))
        step = beishu - 1
        seq_length = self.seq_length * beishu

        x_interpolated = F.interpolate(
            x_transposed,
            size=int(self.seq_length-step),
            mode='linear',
            align_corners=True
        )
        last_slope = x_start_ori[:, -1, :] - x_start_ori[:, -2, :]

        step_percent = 1/beishu
        curentstep = step_percent
        percentlist = []
        for i in range(step):
            percentlist.append(curentstep)
            curentstep = curentstep + step_percent
        percenttensor =torch.tensor(percentlist)
        tail_points = x_start_ori[:, -1:, :] + last_slope.unsqueeze(1) * percenttensor.view(1, step,1).to(x_start_ori.device)

        tail_points_transposed = tail_points.transpose(1, 2)

        x_final = torch.cat([x_interpolated, tail_points_transposed], dim=2)

        x_final = x_final.transpose(1, 2)

        return x_final

    def highrel_out_of_bound_degree(self,highrel_col):
        lower_bound_degree = torch.where(highrel_col < -1, -1 - highrel_col,
                                         torch.tensor(0.0, device=highrel_col.device))
        return lower_bound_degree
    def lowrel_out_of_bound_degree(self,lowrel_col):
        upper_bound_degree = torch.where(lowrel_col > 1, lowrel_col - 1,
                                         torch.tensor(0.0, device=lowrel_col.device))
        return upper_bound_degree
    def closerel_out_of_bound_degree(self,closerel_col):
        upper_bound_degree = torch.where(closerel_col > 1, closerel_col - 1,
                                         torch.tensor(0.0, device=closerel_col.device))
        lower_bound_degree = torch.where(closerel_col < -1, -1 - closerel_col,
                                         torch.tensor(0.0, device=closerel_col.device))
        out_of_bound_degree = upper_bound_degree + lower_bound_degree
        return out_of_bound_degree

    def calculate_out_of_bound_degree_pytorch(self,selected_cols):

        upper_bound_degree = torch.where(selected_cols > 1, selected_cols - 1,
                                         torch.tensor(0.0, device=selected_cols.device))
        lower_bound_degree = torch.where(selected_cols < -1, -1 - selected_cols,
                                         torch.tensor(0.0, device=selected_cols.device))
        out_of_bound_degree = upper_bound_degree + lower_bound_degree
        return out_of_bound_degree

    def diffusion_out(self, x_start_ori, t,traintestflag, target, noise=None, padding_masks=None):

        beishu = int(self.seq_length / len(x_start_ori[0]))
        x_start = self.interpolate_3d_tensor(x_start_ori)
        #x_start = x_start_ori
        target = x_start

        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # noise sample

        model_out = self.output(x, t, padding_masks)
        selected_cols = model_out[:, :, [2, 3, 4]]
        min_value = torch.min(selected_cols)

        highrel_col = model_out[:, :, 2].unsqueeze(2)
        lowrel_col = model_out[:, :, 3].unsqueeze(2)
        closerel_col = model_out[:, :, 4].unsqueeze(2)
        if traintestflag == True:
            count_greater_1 = (selected_cols > 1).sum().item()
            count_less_neg1 = (selected_cols < -1).sum().item()
            self.greater1_total = self.greater1_total + count_greater_1
            self.less_neg1_total = self.less_neg1_total + count_less_neg1




            count_highcross = (highrel_col < -1).sum().item()
            count_lowcross = (lowrel_col > 1).sum().item()
            count_closecrossup = (closerel_col > 1).sum().item()
            count_closecrossdn = (closerel_col < -1).sum().item()

            self.greater1_total2 = self.greater1_total2 + count_lowcross + count_closecrossup
            self.less_neg1_total2 = self.less_neg1_total2 + count_highcross + count_closecrossdn



        train_loss = F.l1_loss(model_out, target, reduction='none')

        model_out_ori = model_out[:, ::beishu, :]
        target_open_col_ori = x_start_ori[..., 0:1]
        model_out_open_col_ori = model_out_ori[..., 0:1]
        train_loss_open_ori = F.l1_loss(model_out_open_col_ori, target_open_col_ori, reduction='none')
        train_loss_ori = F.l1_loss(model_out_ori, x_start_ori, reduction='none')

        target_first_col = target[..., 0]
        model_out_first_col = model_out[..., 0].detach()
        correlations_foropen = []
        correlations_foropen_mini = []

        if traintestflag:
            self.counttest = self.counttest + 1

        target_ori_first_col= x_start_ori[..., 0]
        for i in range(target_first_col.shape[0]):
            batch_target = target_first_col[i]
            batch_model_out = model_out_first_col[i]

            corr_matrix = np.corrcoef(batch_target.cpu().numpy(), batch_model_out.cpu().numpy())


            correlation = corr_matrix[0, 1]

            batch_target_mini = batch_target[::beishu]
            batch_model_out_mini = batch_model_out[::beishu]

            corr_matrix_mini = np.corrcoef(batch_target_mini.cpu().numpy(), batch_model_out_mini.cpu().numpy())
            correlation_mini = corr_matrix_mini[0, 1]

            correlations_foropen.append(correlation)
            correlations_foropen_mini.append(correlation_mini)

            if traintestflag and self.counttest % 10 == 0 and i == 0:
                print("correlation= "+str(correlation))
                print("correlation_mini= "+str(correlation_mini))
                print("batch_target")
                print(batch_target)
                print("batch_model_out")
                print(batch_model_out)
                print("batch_target_mini")
                print(batch_target_mini)
                print("batch_model_out_mini")
                print(batch_model_out_mini)

        aver_corre_foropen = np.mean(correlations_foropen)
        aver_corre_foropen_mini = np.mean(correlations_foropen_mini)

        target_open_col = target[..., 0:1]
        model_out_open_col = model_out[..., 0:1]
        train_loss_open = F.l1_loss(model_out_open_col, target_open_col, reduction='none')

        self.counttrain = self.counttrain + 1



        if self.counttrain % 1000 == 0 or self.counttrain == 1 or traintestflag == True:
            aver_corre_str = "%.2f" % (aver_corre_foropen)
            aver_corre_mini_str = "%.2f" % (aver_corre_foropen_mini)

            lower_0 = 0
            lower_20 = 0
            lower_40 = 0
            lower_60 = 0
            lower_80 = 0
            lower_100 = 0
            for item in correlations_foropen:
                if item <= 0:
                    lower_0 = lower_0 + 1
                elif 0 < item <= 0.2:
                    lower_20 = lower_20 + 1
                elif 0.2< item <= 0.4:
                    lower_40 = lower_40 + 1
                elif 0.4 < item <= 0.6:
                    lower_60 = lower_60 + 1
                elif 0.6 < item <= 0.8:
                    lower_80 = lower_80 + 1
                elif item > 0.8:
                    lower_100 = lower_100 + 1

            if traintestflag:
                if self.testcount_norm.get("aver_corre") is None:
                    self.testcount_norm["aver_corre_total"] = 0
                    self.testcount_norm["aver_corre_count"] = 0
                    self.testcount_norm["aver_corre_mini_total"] = 0
                self.testcount_norm["aver_corre_total"] = self.testcount_norm["aver_corre_total"] + aver_corre_foropen
                self.testcount_norm["aver_corre_mini_total"] = self.testcount_norm["aver_corre_mini_total"] + aver_corre_foropen_mini
                self.testcount_norm["aver_corre_count"] = self.testcount_norm["aver_corre_count"] + 1
                self.testcount_norm["aver_corre"] = self.testcount_norm["aver_corre_total"] / self.testcount_norm["aver_corre_count"]
                self.testcount_norm["aver_corre_mini"] = self.testcount_norm["aver_corre_mini_total"] / self.testcount_norm["aver_corre_count"]

                print("testcount_norm count= " + str(self.testcount_norm["aver_corre_count"]))

                if self.testcount_norm.get("lower_0") is None:
                    self.testcount_norm["lower_0_total"] = 0
                    self.testcount_norm["lower_0_count"] = 0
                self.testcount_norm["lower_0_total"] = self.testcount_norm["lower_0_total"] + lower_0
                self.testcount_norm["lower_0_count"] = self.testcount_norm["lower_0_count"] + 1
                self.testcount_norm["lower_0"] = self.testcount_norm["lower_0_total"] / self.testcount_norm["lower_0_count"]

                if self.testcount_norm.get("lower_20") is None:
                    self.testcount_norm["lower_20_total"] = 0
                    self.testcount_norm["lower_20_count"] = 0
                self.testcount_norm["lower_20_total"] = self.testcount_norm["lower_20_total"] + lower_20
                self.testcount_norm["lower_20_count"] = self.testcount_norm["lower_20_count"] + 1
                self.testcount_norm["lower_20"] = self.testcount_norm["lower_20_total"] / self.testcount_norm["lower_20_count"]

                if self.testcount_norm.get("lower_40") is None:
                    self.testcount_norm["lower_40_total"] = 0
                    self.testcount_norm["lower_40_count"] = 0
                self.testcount_norm["lower_40_total"] = self.testcount_norm["lower_40_total"] + lower_40
                self.testcount_norm["lower_40_count"] = self.testcount_norm["lower_40_count"] + 1
                self.testcount_norm["lower_40"] = self.testcount_norm["lower_40_total"] / self.testcount_norm["lower_40_count"]

                if self.testcount_norm.get("lower_60") is None:
                    self.testcount_norm["lower_60_total"] = 0
                    self.testcount_norm["lower_60_count"] = 0
                self.testcount_norm["lower_60_total"] = self.testcount_norm["lower_60_total"] + lower_60
                self.testcount_norm["lower_60_count"] = self.testcount_norm["lower_60_count"] + 1
                self.testcount_norm["lower_60"] = self.testcount_norm["lower_60_total"] / self.testcount_norm["lower_60_count"]

                if self.testcount_norm.get("lower_80") is None:
                    self.testcount_norm["lower_80_total"] = 0
                    self.testcount_norm["lower_80_count"] = 0
                self.testcount_norm["lower_80_total"] = self.testcount_norm["lower_80_total"] + lower_80
                self.testcount_norm["lower_80_count"] = self.testcount_norm["lower_80_count"] + 1
                self.testcount_norm["lower_80"] = self.testcount_norm["lower_80_total"] / self.testcount_norm["lower_80_count"]

                if self.testcount_norm.get("lower_100") is None:
                    self.testcount_norm["lower_100_total"] = 0
                    self.testcount_norm["lower_100_count"] = 0
                self.testcount_norm["lower_100_total"] = self.testcount_norm["lower_100_total"] + lower_100
                self.testcount_norm["lower_100_count"] = self.testcount_norm["lower_100_count"] + 1
                self.testcount_norm["lower_100"] = self.testcount_norm["lower_100_total"] / self.testcount_norm["lower_100_count"]

            print("norm aver_corre=" + aver_corre_str+";<0="+str(lower_0)+";0~0.2="+str(lower_20)+";0.2~0.4="+str(lower_40)+";0.4~0.6="+str(lower_60)+";0.6~0.8="+str(lower_80)+";>0.8="+str(lower_100))
            print("norm aver_corre_mini= "+aver_corre_mini_str)
            train_loss_open_ori_mean = train_loss_open_ori.mean()
            train_loss_ori_mean = train_loss_ori.mean()
            print("norm train_loss_ori_mean= " + str(train_loss_ori_mean))

        total_loss = train_loss.mean()
        #if self.counttest > 2000:
        overboundary_penalty = self.calculate_out_of_bound_degree_pytorch(selected_cols)
        total_loss = total_loss + overboundary_penalty.mean()

        highoverboundary_penalty = self.highrel_out_of_bound_degree(highrel_col)
        lowoverboundary_penalty = self.highrel_out_of_bound_degree(lowrel_col)
        closeoverboundary_penalty = self.highrel_out_of_bound_degree(closerel_col)
        concatenated_penalty = torch.cat([highoverboundary_penalty,
                                  lowoverboundary_penalty,
                                  closeoverboundary_penalty], dim=2)
        total_penalty = highoverboundary_penalty+lowoverboundary_penalty+closeoverboundary_penalty
        #total_loss = total_loss + total_penalty.mean()

        #total_loss = total_loss + concatenated_penalty.mean()

        if traintestflag == True:
            return train_loss.mean() , train_loss , train_loss_ori
        return total_loss

    def forward(self, x,traintestflag,target, **kwargs):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.diffusion_out(x_start_ori=x, t=t,traintestflag=traintestflag,target=target, **kwargs)

    def return_components(self, x, t: int):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self.q_sample(x, t)
        trend, season, residual = self.model(x, t, return_res=True)
        return trend, season, residual, x

    def fast_sample_infill(self, shape, target, sampling_timesteps, partial_mask=None, clip_denoised=True,
                           model_kwargs=None):
        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='conditional sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise
            img = self.langevin_fn(sample=img, mean=pred_mean, sigma=sigma, t=time_cond,
                                   tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
            target_t = self.q_sample(target, t=time_cond)
            img[partial_mask] = target_t[partial_mask]

        img[partial_mask] = target[partial_mask]

        return img

    def sample_infill(
            self,
            shape,
            target,
            partial_mask=None,
            clip_denoised=True,
            model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='conditional sampling loop time step', total=self.num_timesteps):
            img = self.p_sample_infill(x=img, t=t, clip_denoised=clip_denoised, target=target,
                                       partial_mask=partial_mask, model_kwargs=model_kwargs)

        img[partial_mask] = target[partial_mask]
        return img

    def p_sample_infill(
            self,
            x,
            target,
            t: int,
            partial_mask=None,
            clip_denoised=True,
            model_kwargs=None
    ):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _ = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise

        pred_img = self.langevin_fn(sample=pred_img, mean=model_mean, sigma=sigma, t=batched_times,
                                    tgt_embs=target, partial_mask=partial_mask, **model_kwargs)

        target_t = self.q_sample(target, t=batched_times)
        pred_img[partial_mask] = target_t[partial_mask]

        return pred_img

    def langevin_fn(
            self,
            coef,
            partial_mask,
            tgt_embs,
            learning_rate,
            sample,
            mean,
            sigma,
            t,
            coef_=0.
    ):

        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                x_start = self.output(x=input_embs_param, t=t)

                if sigma.mean() == 0:
                    logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = infill_loss.mean(dim=0).sum()
                else:
                    logp_term = coef * ((mean - input_embs_param) ** 2 / sigma).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = (infill_loss / sigma.mean()).mean(dim=0).sum()

                loss = logp_term + infill_loss
                loss.backward()
                optimizer.step()
                epsilon = torch.randn_like(input_embs_param.data)
                input_embs_param = torch.nn.Parameter(
                    (input_embs_param.data + coef_ * sigma.mean().item() * epsilon).detach())

        sample[~partial_mask] = input_embs_param.data[~partial_mask]
        return sample

    def condition_mean(self, cond_fn, mean, log_variance, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x=x, t=t, **model_kwargs)
        new_mean = (
                mean.float() + torch.exp(log_variance) * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, x_start, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)

        eps = self.predict_noise_from_start(x, t, x_start)
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        pred_xstart = self.predict_start_from_noise(x, t, eps)
        model_mean, _, _ = self.q_posterior(x_start=pred_xstart, x_t=x, t=t)
        return model_mean, pred_xstart

    def sample_cond(
            self,
            shape,
            clip_denoised=True,
            model_kwargs=None,
            cond_fn=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, x_start = self.p_sample(img, t, clip_denoised=clip_denoised, cond_fn=cond_fn,
                                         model_kwargs=model_kwargs)
        return img

    def fast_sample_cond(
            self,
            shape,
            clip_denoised=True,
            model_kwargs=None,
            cond_fn=None
    ):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if cond_fn is not None:
                _, x_start = self.condition_score(cond_fn, x_start, img, time_cond, model_kwargs=model_kwargs)
                pred_noise = self.predict_noise_from_start(img, time_cond, x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img
