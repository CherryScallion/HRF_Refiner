import yaml
import torch
from torch.utils.data import DataLoader
from src.dataset import FMRI_H5_Dataset
from src.network import NeuroRefiner
from src.losses import PhysiologicalLoss
from tqdm import tqdm

def load_config(path='config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 1. Setup
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading H5 from: {cfg['data']['train_h5_path']}")

    # 2. Dataset
    dataset = FMRI_H5_Dataset(
        cfg['data']['train_h5_path'], 
        cfg['data']['window_size']
    )
    loader = DataLoader(
        dataset, 
        batch_size=cfg['train']['batch_size'], 
        shuffle=True, 
        num_workers=cfg['train']['num_workers']
    )

    # 3. Model
    model = NeuroRefiner(cfg).to(device)
    
    # 4. Loss & Optim
    criterion = PhysiologicalLoss(cfg['loss_weights']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['train']['learning_rate']))

    # 5. Loop
    print("Start Training...")
    model.train()
    for epoch in range(cfg['train']['epochs']):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            
            # Forward
            pred = model(x)
            
            # Loss
            loss, details = criterion(pred, y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log
            loop.set_postfix(
                Loss=f"{loss.item():.4f}", 
                CCC=f"{details['ccc']:.3f}", # CCC越低(Loss越低)，相关性越高
                MSE=f"{details['mse']:.4f}"
            )
            
    # Save
    torch.save(model.state_dict(), f"./{cfg['experiment_name']}.pth")
    print("Done.")

if __name__ == "__main__":
    main()