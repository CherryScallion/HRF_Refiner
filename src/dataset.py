import torch
import h5py
import numpy as np
from torch.utils.data import Dataset

class FMRI_H5_Dataset(Dataset):
    def __init__(self, h5_path, window_size):
        """
        [修正版] 支持全长序列预测，自动处理边界填充 (Padding)。
        保证: Input Length == Output Length
        """
        self.h5_path = h5_path
        self.window_size = window_size
        
        # 打开文件获取总长度
        with h5py.File(h5_path, 'r') as f:
            # 假设 dataset 名字是 'pred' 和 'gt'
            # shape: [Total_Timepoints, H, W]
            self.length = f['gt'].shape[0]
            
        # 注意：这里不再减去 window_size
        # 我们要让 idx 覆盖 0 到 length-1 的所有时间点
        self.full_indices = range(self.length)

    def __len__(self):
        return len(self.full_indices)

    def __getitem__(self, idx):
        # 目标：预测第 idx 帧
        # 需要：Base Model 的 [idx - T + 1, ..., idx] 这 T 帧
        
        with h5py.File(self.h5_path, 'r') as f:
            # 1. 获取 Target (当前时刻真实值)
            # Shape: [1, H, W]
            target_slice = f['gt'][idx]
            target_slice = target_slice[np.newaxis, ...] 

            # 2. 获取 Input (历史窗口)
            # 计算切片的起始和结束点
            # 比如 idx=2, T=5. 我们想要 [-2, -1, 0, 1, 2]。
            # 实际能取的是 [0, 1, 2]。前两个要填充。
            
            end_idx = idx + 1
            start_idx = end_idx - self.window_size
            
            if start_idx >= 0:
                # 情况 A: 历史数据充足 (比如 idx=100)
                # 直接切片，不需要填充
                input_chunk = f['pred'][start_idx : end_idx]
            else:
                # 情况 B: 历史数据不足 (比如 idx=0, 1, 2, 3)
                # 这就是你说的"前五个层无法取样"的情况
                
                # a. 先取能取到的部分 (例如 [0, 1, 2])
                valid_chunk = f['pred'][0 : end_idx]
                
                # b. 计算缺多少帧
                pad_len = self.window_size - valid_chunk.shape[0]
                
                # c. 实施填充策略 (Padding Strategy)
                # 策略 1: 零填充 (Zero Padding) - 模型会看到黑色
                # pad_chunk = np.zeros((pad_len, valid_chunk.shape[1], valid_chunk.shape[2]))
                
                # 策略 2: 边缘复制 (Edge Padding) - 推荐！
                # 复制第 0 帧来填充缺失的历史。这样信号是平的，不会突变，模型更容易处理。
                first_frame = valid_chunk[0:1] # shape [1, H, W]
                pad_chunk = np.repeat(first_frame, pad_len, axis=0)
                
                # d. 拼起来
                input_chunk = np.concatenate([pad_chunk, valid_chunk], axis=0)

        # 3. 转换为 Tensor
        # input_chunk shape 必须严格是 [T, H, W]
        input_tensor = torch.from_numpy(input_chunk).float()
        target_tensor = torch.from_numpy(target_slice).float()
        
        return input_tensor, target_tensor