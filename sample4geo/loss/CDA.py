import torch
import torch.nn.functional as F

class CDALoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        """
        CDA Loss 计算跨域一致性损失，包括：
        - 余弦相似度 (Cosine Similarity) 促进全局特征对齐
        - 均方误差 (MSE) 促进局部特征一致性
        参数：
        - alpha: 余弦相似度的权重
        - beta: MSE 损失的权重
        """
        super(CDALoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, f_o_d, f_o_s):
        """
        计算 CDA 损失
        参数：
        - f_o_d: 无人机视角的特征表示 (batch_size, feature_dim)
        - f_o_s: 卫星视角的特征表示 (batch_size, feature_dim)
        返回：
        - CDA 损失值
        """
        # 计算 Cosine 相似度损失（1 - 余弦相似度均值）
        cosine_similarity = F.cosine_similarity(f_o_d, f_o_s, dim=1)
        c_loss = 1 - torch.mean(cosine_similarity)

        # 计算 MSE 损失
        mse_loss = F.mse_loss(f_o_d, f_o_s)

        # 计算加权损失
        loss = self.alpha * c_loss + self.beta * mse_loss
        return loss

