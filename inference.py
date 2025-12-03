import torch
import h5py
import numpy as np
from src.network import NeuroRefiner
from tqdm import tqdm
import yaml

# --- 全局缩放因子 (需与训练时一致) ---
SCALE_FACTOR = 1000.0 

def load_config(path='config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_inference():
    # 1. 初始化
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window_size = cfg['data']['window_size']
    
    # 2. 加载模型
    print("Loading model...")
    model = NeuroRefiner(cfg).to(device)
    # 加载训练好的权重
    checkpoint = torch.load(f"./{cfg['experiment_name']}.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 3. 打开数据 (读模式)
    src_h5 = h5py.File(cfg['data']['val_h5_path'], 'r')
    base_data = src_h5['pred'] # [Total_Time, H, W]
    total_frames = base_data.shape[0]
    h, w = base_data.shape[1], base_data.shape[2]

    # 4. 创建输出文件 (写模式)
    out_h5 = h5py.File(f"./{cfg['experiment_name']}_refined.h5", 'w')
    # 创建 dataset 用于存结果
    refined_ds = out_h5.create_dataset('refined', shape=(total_frames, h, w), dtype='float32')

    print(f"Starting inference on {total_frames} frames...")

    # 5. 逐帧推断 (为了显存安全，Batch Size=1 或者小一点)
    # 这里演示 Batch=1 的逻辑，实际上你可以写个 DataLoader 来加速
    
    with torch.no_grad():
        for i in tqdm(range(total_frames)):
            # --- 构造 Input (复用 dataset.py 的逻辑) ---
            end_idx = i + 1
            start_idx = end_idx - window_size
            
            if start_idx >= 0:
                input_chunk = base_data[start_idx : end_idx]
            else:
                # Padding 逻辑
                valid_chunk = base_data[0 : end_idx]
                pad_len = window_size - valid_chunk.shape[0]
                first_frame = valid_chunk[0:1]
                pad_chunk = np.repeat(first_frame, pad_len, axis=0)
                input_chunk = np.concatenate([pad_chunk, valid_chunk], axis=0)

            # 转 Tensor & 缩放
            input_tensor = torch.from_numpy(input_chunk).float().to(device)
            input_tensor = input_tensor.unsqueeze(0) / SCALE_FACTOR # [1, T, H, W]

            # 推理
            # Output: [1, 1, H, W]
            refined_tensor = model(input_tensor)

            # 还原缩放 & 存入 H5
            refined_npy = refined_tensor.squeeze().cpu().numpy() * SCALE_FACTOR
            refined_ds[i] = refined_npy

    # 6. 清理
    src_h5.close()
    out_h5.close()
    print(f"Inference done. Saved to ./{cfg['experiment_name']}_refined.h5")

if __name__ == "__main__":
    run_inference()
