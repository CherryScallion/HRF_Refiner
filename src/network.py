import torch
import torch.nn as nn

class NeuroRefiner(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        T = config['data']['window_size']
        C = config['model']['hidden_dim']
        
        # ============================================================
        # [核心组件 1] 时域卷积核 (The Kernel Movement)
        # ============================================================
        # 你的工作重心在这里：
        # 虽然是 Conv2d(k=1)，但因为输入通道是 T (时间)，
        # 所以对于空间点 (x,y)，它执行的是：
        # Out(x,y) = w_0 * In(t) + w_1 * In(t-1) + ... + w_4 * In(t-4)
        # 这是一个纯粹的、可学习的时域滤波器 (FIR Filter)。
        self.temporal_kernel = nn.Conv2d(
            in_channels=T, 
            out_channels=C, 
            kernel_size=1, 
            padding=0,
            bias=True
        )

        # ============================================================
        # [核心组件 2] 空间约束 (不改变时间，只整合空间)
        # ============================================================
        self.spatial_body = nn.Sequential(
            # 使用膨胀卷积捕捉局部脑区连接
            nn.Conv2d(C, C, 3, padding=1, dilation=1),
            nn.GroupNorm(8, C),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(C, C, 3, padding=2, dilation=2),
            nn.GroupNorm(8, C),
            nn.SiLU(inplace=True)
        )

        # ============================================================
        # [核心组件 3] 残差投影
        # ============================================================
        self.projector = nn.Conv2d(C, 1, 1)

        # 零初始化：保证初始 Refined = Input
        nn.init.constant_(self.projector.weight, 0)
        nn.init.constant_(self.projector.bias, 0)

    def forward(self, x):
        """
        x: [Batch, T, H, W] - 包含 Base Model 的过去 T 帧输出
        """
        # 1. 取出 Base Model 对当前的预测 (作为基准)
        base_current = x[:, -1:, :, :]

        # 2. 卷积核运动：融合时域信息
        feat = self.temporal_kernel(x)
        
        # 3. 空间整合
        feat = self.spatial_body(feat)
        
        # 4. 计算残差
        residual = self.projector(feat)
        
        # 5. 修正
        return base_current + residual