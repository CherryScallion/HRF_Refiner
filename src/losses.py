import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysiologicalLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.w_mse, self.w_ssim, self.w_ccc = weights
        self.mse_fn = nn.MSELoss()

    def _ccc_loss(self, y_pred, y_true):
        """
        Lin's Concordance Correlation Coefficient
        兼顾：相关性(Shape) + 变异性(Scale) + 偏差(Location)
        """
        # 拉平: [B, 1, H, W] -> [B, H*W]
        pred_flat = y_pred.view(y_pred.size(0), -1)
        true_flat = y_true.view(y_true.size(0), -1)

        mean_pred = torch.mean(pred_flat, 1)
        mean_true = torch.mean(true_flat, 1)
        
        var_pred = torch.var(pred_flat, 1)
        var_true = torch.var(true_flat, 1)
        
        std_pred = torch.std(pred_flat, 1)
        std_true = torch.std(true_flat, 1)
        
        # Covariance
        vx = pred_flat - mean_pred.unsqueeze(1)
        vy = true_flat - mean_true.unsqueeze(1)
        covariance = torch.sum(vx * vy, 1) / (pred_flat.size(1) - 1)
        
        # Pearson Correlation
        pcc = covariance / (std_pred * std_true + 1e-8)
        
        # CCC
        numerator = 2 * pcc * std_pred * std_true
        denominator = var_pred + var_true + (mean_pred - mean_true)**2 + 1e-8
        
        return 1 - torch.mean(numerator / denominator)

    def _ssim_loss(self, img1, img2):
        # 简化的 SSIM 实现，关注局部结构
        # 这里为了工程简洁，使用平均池化近似高斯窗
        mu1 = F.avg_pool2d(img1, 3, 1, 1)
        mu2 = F.avg_pool2d(img2, 3, 1, 1)
        
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1**2, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2**2, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()

    def forward(self, y_pred, y_true):
        l_mse = self.mse_fn(y_pred, y_true)
        l_ssim = self._ssim_loss(y_pred, y_true)
        l_ccc = self._ccc_loss(y_pred, y_true)
        
        total = self.w_mse * l_mse + self.w_ssim * l_ssim + self.w_ccc * l_ccc
        return total, {"mse": l_mse.item(), "ssim": l_ssim.item(), "ccc": l_ccc.item()}