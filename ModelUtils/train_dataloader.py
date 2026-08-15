import torch
from ModelUtils.Crdatasets import CustomDataset
from Utils.io_utils import instantiate_from_config


class Train_DataLoader(object):
    def __init__(self,config,data_withralship=None):
        self.batch_size = config['train_dataload']['batch_size']
        self.shuffle = config['train_dataload']['shuffle']
        self.drop_last = config['train_dataload']['drop_last']

        self.dataset = CustomDataset(
            name=config['train_dataload']['name'],
            data_root=config['train_dataload']['data_root'],
            window=config['train_dataload']['window'],
            period='train',
            data_withrelaship=data_withralship
        )


    def getDataLoader(self):
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=self.shuffle,
                                                 num_workers=0,
                                                 pin_memory=True,
                                                 sampler=None,
                                                 drop_last=self.drop_last)
        return dataloader,self.dataset


