import torch
import torch.nn as nn
import torchmetrics

class CCCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(CCCLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Stats
        mu_pred = torch.mean(pred_flat, dim=1)
        mu_target = torch.mean(target_flat, dim=1)
        var_pred = torch.var(pred_flat, dim=1, unbiased=False)
        var_target = torch.var(target_flat, dim=1, unbiased=False)
        std_pred = torch.sqrt(var_pred + self.eps)
        std_target = torch.sqrt(var_target + self.eps)

        # Covariance & PCC
        covar = torch.mean((pred_flat - mu_pred.unsqueeze(1)) * (target_flat - mu_target.unsqueeze(1)), dim=1)
        pcc = covar / (std_pred * std_target + self.eps)

        # CCC
        numerator = 2 * pcc * std_pred * std_target
        denominator = var_pred + var_target + (mu_pred - mu_target)**2 + self.eps
        ccc = numerator / denominator

        # 返回 1-CCC 作为 Loss，同时也返回 CCC 的平均值用于监控
        return 1.0 - torch.mean(ccc), torch.mean(ccc)

class ssim_mse_ccc_loss(nn.Module):
    def __init__(self, lambda_ssim=1.0, lambda_mse=10.0, lambda_ccc=1.0):
        super(ssim_mse_ccc_loss, self).__init__()
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)
        self.mse = nn.MSELoss()
        self.ccc = CCCLoss()
        
        self.lambda_ssim = lambda_ssim
        self.lambda_mse = lambda_mse
        self.lambda_ccc = lambda_ccc

    def forward(self, img1, img2):
        # 计算各分项
        ssim_val = self.ssim(img1, img2)
        loss_ssim = 1.0 - ssim_val
        
        loss_mse = self.mse(img1, img2)
        
        loss_ccc, ccc_val = self.ccc(img1, img2) # 接收 CCC 值
        
        # 总 Loss
        total_loss = (self.lambda_ssim * loss_ssim + 
                      self.lambda_mse * loss_mse + 
                      self.lambda_ccc * loss_ccc)
        
        # 打包返回: 总Loss, 以及各指标字典
        metrics = {
            "ssim": ssim_val,
            "mse": loss_mse,
            "ccc": ccc_val
        }
        return total_loss, metrics
    
    
# 保留旧的类以防兼容性问题，或者直接替换
class GANLoss(nn.Module):
    # ... (保持不变) ...
    def __init__(self, real_label=1.0, fake_label=0.0):
        super(GANLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss