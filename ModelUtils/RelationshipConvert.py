import os
import torch
import numpy as np
import pandas as pd




class RelationshipConvert(object):
    def __init__(self,config):
        self.dataroot = config['train_dataload']['data_root']
        self.volumepos = config['train_dataload']['volumepos']
        self.config = config


    def resdata(self,data):
        open_price = data[:,:, 0]
        high_relaship = data[:,:, 2]
        low_relaship = data[:,:, 3]
        close_relaship = data[:,:, 4]
        high_price = (1+high_relaship)*open_price
        low_price = (1+low_relaship)*open_price
        close_price = close_relaship*(high_price-low_price)+low_price
        if self.volumepos != -1:
            volume = data[:,:, 8]
            result = np.stack([
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            ], axis=2)
        else:
            result = np.stack([
                open_price,
                high_price,
                low_price,
                close_price
            ], axis=2)
        return result

    def loaddata(self):
        df = pd.read_csv(self.dataroot, header=0)
        data = df.values

        open_price = data[:, 0]
        high_price = data[:, 1]
        low_price = data[:, 2]
        close_price = data[:, 3]


        high_relaship = (high_price - open_price) / open_price
        low_relaship = (low_price - open_price) / open_price
        high_low_diff = high_price - low_price
        close_relaship = np.where(high_low_diff != 0,
                                  (close_price - low_price) / high_low_diff,
                                  np.nan)
        open_price_change = np.zeros_like(open_price)
        open_price_change[1:] = (open_price[1:] - open_price[:-1]) / open_price[:-1]
        high_price_change = np.zeros_like(high_price)
        high_price_change[1:] = (high_price[1:] - high_price[:-1]) / high_price[:-1]
        low_price_change = np.zeros_like(low_price)
        low_price_change[1:] = (low_price[1:] - low_price[:-1]) / low_price[:-1]
        close_price_change = np.zeros_like(close_price)
        close_price_change[1:] = (close_price[1:] - close_price[:-1]) / close_price[:-1]


        if self.volumepos != -1:
            volume = data[:, self.volumepos]
            n = len(volume)
            volume_normalized = np.zeros_like(volume, dtype=np.float64)

            for i in range(250, n):
                window_volume = volume[i - 250:i]
                volume_mean = np.mean(window_volume)
                volume_std = np.std(window_volume)

                if volume_std == 0:
                    volume_normalized[i] = 0
                else:
                    volume_normalized[i] = (volume[i] - volume_mean) / volume_std
            result = np.column_stack([
                open_price,
                open_price_change,
                high_relaship,
                low_relaship,
                close_relaship,
                high_price_change,
                low_price_change,
                close_price_change,
                volume_normalized,
                high_price,
                low_price,
                close_price
            ])
            result = result[250:]
        else:
            print("no volume")
            result = np.column_stack([
                open_price,
                open_price_change,
                high_relaship,
                low_relaship,
                close_relaship,
                high_price_change,
                low_price_change,
                close_price_change,
                high_price,
                low_price,
                close_price
            ])
        return result

class RelationshipConvert_SignleClose(object):
    def __init__(self,config):
        self.dataroot = config['train_dataload']['data_root']
        self.volumepos = config['train_dataload']['volumepos']
        self.config = config


    def resdata(self,data):
        open_price = data[:,:, 0]
        high_relaship = data[:,:, 2]
        low_relaship = data[:,:, 3]
        close_relaship = data[:,:, 4]
        high_price = (1+high_relaship)*open_price
        low_price = (1+low_relaship)*open_price
        close_price = (1+close_relaship)*open_price
        if self.volumepos != -1:
            volume = data[:,:, 8]
            result = np.stack([
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            ], axis=2)
        else:
            result = np.stack([
                open_price,
                high_price,
                low_price,
                close_price
            ], axis=2)
        return result

    def loaddata(self):
        df = pd.read_csv(self.dataroot, header=0)
        data = df.values

        open_price = data[:, 0]
        high_price = data[:, 1]
        low_price = data[:, 2]
        close_price = data[:, 3]


        high_relaship = (high_price - open_price) / open_price
        low_relaship = (low_price - open_price) / open_price
        close_relaship = (close_price - open_price) / open_price
        open_price_change = np.zeros_like(open_price)
        open_price_change[1:] = (open_price[1:] - open_price[:-1]) / open_price[:-1]
        high_price_change = np.zeros_like(high_price)
        high_price_change[1:] = (high_price[1:] - high_price[:-1]) / high_price[:-1]
        low_price_change = np.zeros_like(low_price)
        low_price_change[1:] = (low_price[1:] - low_price[:-1]) / low_price[:-1]
        close_price_change = np.zeros_like(close_price)
        close_price_change[1:] = (close_price[1:] - close_price[:-1]) / close_price[:-1]


        if self.volumepos != -1:
            volume = data[:, self.volumepos]
            n = len(volume)
            volume_normalized = np.zeros_like(volume, dtype=np.float64)

            for i in range(250, n):
                window_volume = volume[i - 250:i]
                volume_mean = np.mean(window_volume)
                volume_std = np.std(window_volume)

                if volume_std == 0:
                    volume_normalized[i] = 0
                else:
                    volume_normalized[i] = (volume[i] - volume_mean) / volume_std
            result = np.column_stack([
                open_price,
                open_price_change,
                high_relaship,
                low_relaship,
                close_relaship,
                high_price_change,
                low_price_change,
                close_price_change,
                volume_normalized,
                high_price,
                low_price,
                close_price
            ])
            result = result[250:]
        else:
            print("no volume")
            result = np.column_stack([
                open_price,
                open_price_change,
                high_relaship,
                low_relaship,
                close_relaship,
                high_price_change,
                low_price_change,
                close_price_change,
                high_price,
                low_price,
                close_price
            ])
        return result