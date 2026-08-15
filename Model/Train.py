import os
import sys
import torch
import numpy as np

from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import pandas as pd
from ModelUtils.ReduceLROnPlateauWithWarmup import ReduceLROnPlateauWithWarmup

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))



class Trainer(object):
    def __init__(self, config, model, dataloader, dataloader_test=None, logger=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.lr_anneal_steps = config['net']['lr_anneal_steps']
        self.dataloader = dataloader['dataloader']
        self.step = 0
        self.dataloader_test = dataloader_test['dataloader']
        start_lr = config['net'].get('base_lr', 1.0e-4)
        ema_decay = config['net']['decay']
        ema_update_every = config['net']['update_interval']
        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)
        self.sch = ReduceLROnPlateauWithWarmup(
            optimizer = self.opt,
            factor = config['net']['scheduler']['factor'],
            patience = config['net']['scheduler']['patience'],
            threshold = config['net']['scheduler']['threshold'],
            threshold_mode = config['net']['scheduler']['threshold_mode'],
            min_lr = config['net']['scheduler']['min_lr'],
            verbose = config['net']['scheduler']['verbose'],
            warmup_lr = config['net']['scheduler']['warmup_lr'],
            warmup = config['net']['scheduler']['warmup']
        )


    def test_traindata(self):
        device = self.device
        self.model.eval()
        print("test_traindata ****************")
        countbatch = 0
        traintestflag = True
        all_batch_means = []
        all_batch_means_ori = []
        for data_train in self.dataloader:
            data_train = data_train.to(device)
            loss, losstensor, losstensor_ori = self.model(data_train, traintestflag, target=data_train)

            loss_means = losstensor.mean(dim=[1, 2])
            all_batch_means.append(loss_means.detach().cpu().numpy())

            loss_means_ori = losstensor_ori.mean(dim=[1, 2])
            all_batch_means_ori.append(loss_means_ori.detach().cpu().numpy())

            countbatch = countbatch + 1

        all_means = np.concatenate(all_batch_means)
        all_means_ori = np.concatenate(all_batch_means_ori)
        df = pd.DataFrame({
            'sample_index': range(len(all_means)),
            'loss_mean': all_means
        })
        df_ori = pd.DataFrame({
            'sample_index': range(len(all_means_ori)),
            'loss_mean': all_means_ori
        })

        csv_filename = 'sample_loss_means.csv'
        df.to_csv(csv_filename, index=False)
        print(f"loss mean saved to: {csv_filename}")

        print(f"total sample: {len(all_means)}")
        print(f"loss mean: {all_means.mean():.4f}")
        print(f"loss std: {all_means.std():.4f}")
        print(f"min loss: {all_means.min():.4f}")
        print(f"max loss: {all_means.max():.4f}")

        print(f"total sample ori: {len(all_means_ori)}")
        print(f"loss mean ori: {all_means_ori.mean():.4f}")
        print(f"loss std ori: {all_means_ori.std():.4f}")
        print(f"min loss ori: {all_means_ori.min():.4f}")
        print(f"max loss ori: {all_means_ori.max():.4f}")

        print("countbatch= " + str(countbatch))
        print("test_traindata over")
        print(loss)
        print("Test Result*****************")
        self.model.test_result()

    def test_testdata(self):
        device = self.device
        self.model.eval()
        print("test_testdata ****************")
        countbatch = 0
        traintestflag = True
        all_batch_means = []
        all_batch_means_ori = []
        for data_train in self.dataloader_test:
            data_train = data_train.to(device)
            loss, losstensor, losstensor_ori = self.model(data_train, traintestflag, target=data_train)

            loss_means = losstensor.mean(dim=[1, 2])
            all_batch_means.append(loss_means.detach().cpu().numpy())

            loss_means_ori = losstensor_ori.mean(dim=[1, 2])
            all_batch_means_ori.append(loss_means_ori.detach().cpu().numpy())

            countbatch = countbatch + 1

        all_means = np.concatenate(all_batch_means)
        all_means_ori = np.concatenate(all_batch_means_ori)
        df = pd.DataFrame({
            'sample_index': range(len(all_means)),
            'loss_mean': all_means
        })
        df_ori = pd.DataFrame({
            'sample_index': range(len(all_means_ori)),
            'loss_mean': all_means_ori
        })

        csv_filename = 'sample_loss_means.csv'
        df.to_csv(csv_filename, index=False)
        print(f"loss mean saved to: {csv_filename}")

        print(f"total sample: {len(all_means)}")
        print(f"loss mean: {all_means.mean():.4f}")
        print(f"loss std: {all_means.std():.4f}")
        print(f"min loss: {all_means.min():.4f}")
        print(f"max loss: {all_means.max():.4f}")

        print(f"total sample ori: {len(all_means_ori)}")
        print(f"loss mean ori: {all_means_ori.mean():.4f}")
        print(f"loss std ori: {all_means_ori.std():.4f}")
        print(f"min loss ori: {all_means_ori.min():.4f}")
        print(f"max loss ori: {all_means_ori.max():.4f}")

        print("countbatch= " + str(countbatch))
        print("test_testdata over")
        print(loss)
        print("Test Result*****************")
        self.model.test_result()

    def train(self):
        print("start train function")
        with tqdm(total=self.lr_anneal_steps) as pbar:
            while self.step < self.lr_anneal_steps:

                data = next(self.next_batch()).to(self.model.betas.device)
                traintestflag = False
                loss = self.model(data, traintestflag, target=data)

                loss = loss
                loss.backward()
                total_loss = loss.item()

                loss_str = f"l1 loss: {total_loss:.6f}"
                pbar.set_description(loss_str)

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                self.ema.update()

                pbar.update(1)

        print('training complete')

    def next_batch(self):
        """
        Get the next batch of data.
        """
        while True:
            for data in self.dataloader:
                yield data

    def sample(self, num_cycle, size_every,seq_len,aug_times,feature_nums, model_kwargs=None, cond_fn=None):
        samples = np.empty([0, seq_len*aug_times, feature_nums])
        count = 0
        for _ in range(num_cycle):
            sample = self.ema.ema_model.generate_mts(batch_size=size_every, model_kwargs=model_kwargs, cond_fn=cond_fn)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()
            count = count + 1

        cut_samples = samples[:, ::aug_times, :]
        return cut_samples




