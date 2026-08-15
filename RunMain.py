import os
import torch
import numpy as np
import yaml
from ModelUtils.train_dataloader import Train_DataLoader
from Model.GDiffusion import RFC_Diff
from Model.Train import Trainer
from ModelUtils.RelationshipConvert import RelationshipConvert,RelationshipConvert_SignleClose

ConfigPath = './Config/cryptocurrency_BTCUSD_DFHC.yaml'
save_dir = './Fake_Sample'
with open(ConfigPath) as f:
    Config = yaml.full_load(f)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

if Config['train_dataload']['name'] == "ETTH" or Config['train_dataload']['name'] == "energy":
    train_loader = Train_DataLoader(Config)
    dataloader, dataset = train_loader.getDataLoader()
else:
    relationshipconvert = RelationshipConvert(Config)
    #relationshipconvert = RelationshipConvert_SignleClose(Config)
    data_withRel = relationshipconvert.loaddata()
    train_loader = Train_DataLoader(Config,data_withRel)
    dataloader,dataset = train_loader.getDataLoader()

diffusion_model = RFC_Diff(
        seq_length=Config["net"]["seq_ori_len"]*Config["aug"]["aug_times"],
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
        aug_times=Config["aug"]["aug_times"],
        kernel_size=Config["aug"]["kernel_size"]
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

fake_data_final_groundtruth = relationshipconvert.resdata(fake_data_groundtruth)



np.save(os.path.join(save_dir, f'ddpm_fake_Groudtruth_stocks.npy'), fake_data_groundtruth)