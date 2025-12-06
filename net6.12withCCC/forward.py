# 文件名: forward.py
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import os

from models import create_unet, EEGEncoder, fMRIDecoder, EEG2fMRINet

# ================= 配置 =================
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path(r"D:\推特数据爬取\E2fNet-main\Data\H5_Unified")
OUTPUT_ROOT = Path(r"D:\推特数据爬取\E2fNet-main\Output")
MODEL_PATH = Path(r"D:\推特数据爬取\E2fNet-main\pre\latest_Unified.pth")
FMRI_CHANNEL = 32

OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)

# ================= 辅助函数 =================
def get_normalization_stats(data_root):
    # (同 test.py，为了独立运行重复一遍)
    train_files = sorted([f.name for f in data_root.glob("Unified_Train_*.h5")])
    with h5py.File(data_root / train_files[0], 'r') as f:
        eeg_sample = f['eeg'][:]
        return float(eeg_sample.min()), float(eeg_sample.max())

# ================= 主流程 =================
def run_inference():
    min_eeg, max_eeg = get_normalization_stats(DATA_ROOT)
    
    print("加载模型...")
    eeg_encoder = EEGEncoder(in_channels=20, img_size=64)
    unet_module = create_unet(in_channels=256, out_channels=256)
    fmri_decoder = fMRIDecoder(in_channels=256, out_channels=FMRI_CHANNEL)
    model = EEG2fMRINet(eeg_encoder, unet_module, fmri_decoder).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_files = sorted([p.name for p in DATA_ROOT.glob("*.h5")])
    
    print(f"开始生成预测，结果将保存至: {OUTPUT_ROOT}")
    
    with torch.no_grad():
        for file_name in all_files:
            file_path = DATA_ROOT / file_name
            output_path = OUTPUT_ROOT / f"Pred_{file_name}"
            
            # 读取
            with h5py.File(file_path, 'r') as f:
                eeg_raw = f['eeg'][:]
                fmri_gt = f['fmri'][:]
            
            # 预处理
            eeg_norm = (eeg_raw - min_eeg) / (max_eeg - min_eeg + 1e-10)
            eeg_tensor = torch.from_numpy(eeg_norm.astype(np.float32))
            
            # 推理 (按 Batch 进行，防止显存溢出)
            loader = DataLoader(eeg_tensor, batch_size=BATCH_SIZE, shuffle=False)
            pred_list = []
            
            for batch_eeg in tqdm(loader, desc=f"Processing {file_name}"):
                batch_eeg = batch_eeg.to(DEVICE)
                batch_pred = model(batch_eeg)
                pred_list.append(batch_pred.cpu().numpy())
            
            # 拼接结果
            pred_fmri = np.concatenate(pred_list, axis=0)
            
            # 保存到 H5
            with h5py.File(output_path, 'w') as hf:
                # 保存预测值
                hf.create_dataset('pred_fmri', data=pred_fmri, compression='gzip')
                # 同时保存真实值方便对比 (可选)
                
            print(f"已保存: {output_path} (Shape: {pred_fmri.shape})")

if __name__ == "__main__":
    run_inference()