# 文件名: debug_tccc_visualize.py
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from models import create_unet, EEGEncoder, fMRIDecoder, EEG2fMRINet

# ================= 配置 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path(r"D:\推特数据爬取\E2fNet-main\Data\H5_Unified")
MODEL_PATH = Path(r"D:\推特数据爬取\E2fNet-main\pre\latest_Unified.pth") 
# 使用 Test Chunk 0
TEST_FILE = DATA_ROOT / "Unified_Test_Chunk_0.h5"

def visualize_time_course():
    print(f"Loading Test File: {TEST_FILE}")
    with h5py.File(TEST_FILE, 'r') as f:
        eeg_raw = f['eeg'][:]
        fmri_gt = f['fmri'][:]
    
    # 获取归一化参数 (粗略)
    min_eeg, max_eeg = eeg_raw.min(), eeg_raw.max()
    eeg_norm = (eeg_raw - min_eeg) / (max_eeg - min_eeg + 1e-10)
    
    # 加载模型
    print("Loading Model...")
    eeg_encoder = EEGEncoder(in_channels=20, img_size=64)
    unet_module = create_unet(in_channels=256, out_channels=256)
    fmri_decoder = fMRIDecoder(in_channels=256, out_channels=32)
    model = EEG2fMRINet(eeg_encoder, unet_module, fmri_decoder).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 推理 (取前 200 个时间点，方便画图)
    time_steps = 200
    if len(eeg_norm) < time_steps: time_steps = len(eeg_norm)
    
    input_tensor = torch.from_numpy(eeg_norm[:time_steps].astype(np.float32)).to(DEVICE)
    
    print("Running Inference...")
    with torch.no_grad():
        # 分批跑防止爆显存
        pred_list = []
        for i in range(0, time_steps, 32):
            batch = input_tensor[i:i+32]
            pred = model(batch)
            pred_list.append(pred.cpu().numpy())
        pred_fmri = np.concatenate(pred_list, axis=0) # [200, 32, 64, 64]
        gt_fmri = fmri_gt[:time_steps]
        
    # --- 诊断核心：找几个活跃点画出来 ---
    
    # 1. 找到 GT 变异最大的体素 (最活跃的脑区)
    # 计算每个体素的时间方差
    gt_var = np.var(gt_fmri, axis=0) # [32, 64, 64]
    
    # 展平找索引
    flat_indices = np.argsort(gt_var.flatten())[::-1] # 降序
    top_indices = flat_indices[:5] # 取最活跃的 5 个点
    
    plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(top_indices):
        # 反解坐标
        c, h, w = np.unravel_index(idx, (32, 64, 64))
        
        # 提取曲线
        gt_curve = gt_fmri[:, c, h, w]
        pred_curve = pred_fmri[:, c, h, w]
        
        # 统计数据
        gt_std = np.std(gt_curve)
        pred_std = np.std(pred_curve)
        correlation = np.corrcoef(gt_curve, pred_curve)[0, 1]
        
        plt.subplot(5, 1, i+1)
        plt.plot(gt_curve, label='Ground Truth', color='black', alpha=0.7)
        plt.plot(pred_curve, label=f'Prediction (Corr={correlation:.2f})', color='red', linewidth=2)
        plt.title(f"Voxel [{c},{h},{w}] - GT_Std: {gt_std:.4f} | Pred_Std: {pred_std:.4f}")
        plt.legend(loc='upper right')
        
    plt.tight_layout()
    plt.savefig('debug_time_course.png')
    print("图表已保存为 debug_time_course.png，请查看！")
    
    # 打印数值诊断
    print("\n=== 数值诊断 ===")
    print(f"GT 平均标准差 (全脑): {np.mean(np.std(gt_fmri, axis=0)):.6f}")
    print(f"Pred 平均标准差 (全脑): {np.mean(np.std(pred_fmri, axis=0)):.6f}")
    ratio = np.mean(np.std(pred_fmri, axis=0)) / np.mean(np.std(gt_fmri, axis=0))
    print(f"方差比率 (Pred/GT): {ratio:.4f}")
    
    if ratio < 0.1:
        print("🚨 结论: 模型坍塌 (Model Collapse)。预测值几乎不动，只输出了平均图像。")
    elif correlation < 0.1:
        print("🚨 结论: 随机波动。模型有输出波动，但和真实值完全没关系。")
    else:
        print("✅ 结论: 看起来还行？那可能是之前 TCCC 脚本计算有误。")

if __name__ == "__main__":
    visualize_time_course()