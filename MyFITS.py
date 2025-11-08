import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in

        # 直接使用cut_freq作为主导频率
        self.dominance_freq = configs.cut_freq
        # 计算长度比例用于能量补偿
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        # 使用复数线性层进行频率上采样
        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat))
        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        
        # RIN标准化
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        # 计算FFT并应用低通滤波器
        low_specx = torch.fft.rfft(x, dim=1)
        # 只保留主导频率以下的部分
        low_specx = low_specx[:, 0:self.dominance_freq, :]

        # 频率上采样
        if self.individual:
            low_specxy_ = torch.zeros([low_specx.size(0), int(self.dominance_freq * self.length_ratio), low_specx.size(2)], 
                                     dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:, :, i] = self.freq_upsampler[i](low_specx[:, :, i].permute(0, 1)).permute(0, 1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)

        # 零填充到所需长度
        low_specxy = torch.zeros([low_specxy_.size(0), int((self.seq_len + self.pred_len) / 2 + 1), low_specxy_.size(2)], 
                                dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_

        # IFFT变换回时域
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        
        # 能量补偿（
        low_xy = low_xy * self.length_ratio

        # 反标准化
        xy = low_xy * torch.sqrt(x_var) + x_mean
        low_xy_output = low_xy * torch.sqrt(x_var)

        return xy, low_xy_output

