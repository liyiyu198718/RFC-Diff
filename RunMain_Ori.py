import os
import torch
import numpy as np
import yaml
from ModelUtils.train_dataloader import Train_DataLoader
from Model.GDiffusion_forOri import RFC_Diff
from Model.Train import Trainer
from ModelUtils.RelationshipConvert import RelationshipConvert
from ModelUtils.SampleCrossLine import CountCrossLine

ConfigPath = './Config/stocks_M000001_DFHC_ori.yaml'
with open(ConfigPath) as f:
    Config = yaml.full_load(f)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


train_loader = Train_DataLoader(Config)
dataloader,dataset = train_loader.getDataLoader()

diffusion_model = RFC_Diff(
        seq_length=Config["net"]["seq_ori_len"],
        feature_size=Config["net"]["feature_ori_nums"],
        n_layer_enc=Config["net"]["n_layer_enc"],
        n_layer_dec=Config["net"]["n_layer_dec"],
        d_model=Config["net"]["d_model"],
        timesteps=Config["net"]["timesteps"],
        sampling_timesteps=Config["net"]["sampling_timesteps"],
        beta_schedule=Config["net"]["beta_schedule"],
        n_heads=Config["net"]["n_heads"],
        mlp_hidden_times=Config["net"]["mlp_hidden_times"],
        attn_pd=Config["net"]["attn_pd"],
        resid_pd=Config["net"]["resid_pd"],
        dataset=dataset,
    ).to(device)

dl_info = {
        'dataloader': dataloader,
        'dataset': dataset
    }

trainer = Trainer(config=Config, model=diffusion_model, dataloader=dl_info,dataloader_test=dl_info)
print("initial Trainer Over")
print("Train Start")
trainer.train()
trainer.test_traindata()

fake_data = trainer.sample(2, size_every=2001, seq_len=Config["net"]["seq_ori_len"],aug_times=Config["aug"]["aug_times"], feature_nums=Config["net"]["feature_ori_nums"])
fake_data_groundtruth = dataset.unnormalize(fake_data)


openhigh_error,openlow_error,closehigh_error,closelow_error,highlow_error = CountCrossLine(fake_data_groundtruth)
