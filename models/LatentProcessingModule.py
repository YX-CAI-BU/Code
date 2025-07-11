import torch
import torch.nn as nn
import torch.nn.functional as F

def sample_latent(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def gumbel_sigmoid(logits, tau=1.0, eps=1e-10):
    """ Gumbel-Sigmoid Approximation """
    U = torch.rand_like(logits)
    g = -torch.log(-torch.log(U + eps) + eps)
    return torch.sigmoid((logits + g) / tau)

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class LatentProcessingModule(nn.Module):
    def __init__(self, dim_in=128, dim_latent=256, seq_len=8, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim_in = dim_in
        self.dim_latent = dim_latent
        self.seq_len = seq_len

        # 1. 输入特征投影 + LayerNorm + Dropout + 噪声
        self.input_proj = nn.Sequential(
            nn.Linear(dim_in, dim_latent),
            nn.LayerNorm(dim_latent),
            GaussianNoise(std=0.1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. Anchor tokens（共享结构引导 token）
        self.shared_anchor_tokens = nn.Parameter(torch.randn(4, dim_latent))  # WRIST, PALM, FINGER_ROOT, FINGER_TIP

        # 3. Structure-Aware Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_latent, nhead=num_heads, batch_first=True, dropout=dropout
        )
        self.st_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4. Gumbel-Softmax 控制 latent 启用
        self.gumbel_gate = nn.Sequential(
            nn.Linear(dim_latent, 1),
            nn.Sigmoid()  # 可切换为 Gumbel-Sigmoid
        )

        # 5. 输出 mu / logvar，左右手分路
        self.left_mu = nn.Linear(dim_latent * 2, dim_latent)
        self.left_logvar = nn.Linear(dim_latent * 2, dim_latent)
        self.right_mu = nn.Linear(dim_latent * 2, dim_latent)
        self.right_logvar = nn.Linear(dim_latent * 2, dim_latent)

        # 用于骨结构对齐后的 z 融合输出（可选）
        self.fusion_proj = nn.Sequential(
            nn.Linear(dim_latent * 2, dim_latent),
            nn.ReLU(),
            nn.LayerNorm(dim_latent)
        )

    def forward(self, fused_feats):  # fused_feats: list of [B, T, C] from fusion stages
        """
        Args:
            fused_feats: list of [B, T, dim_in], from each stage
        Returns:
            z_corr: [B, dim_latent], final fused latent
            mu_all, logvar_all: List of per-stage [B, dim_latent, T] for structure supervision
            mu_L/R, logvar_L/R: [B, dim_latent]
        """
        mu_all, logvar_all = [], []

        for stage_idx, feats in enumerate(fused_feats):
            B, T, C = feats.shape

            # (1) Input Projection with Gaussian Noise
            x = self.input_proj(feats)  # [B, T, dim_latent]

            # (2) Insert shared anchor tokens
            anchors = self.shared_anchor_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, 4, dim_latent]
            x_with_anchor = torch.cat([anchors, x], dim=1)  # [B, 4+T, dim_latent]

            # (3) ST-Transformer
            x_enc = self.st_transformer(x_with_anchor)  # [B, 4+T, dim_latent]

            # (4) Take only the last T tokens (skip anchors for latent extraction)
            latent_seq = x_enc[:, 4:]  # [B, T, dim_latent]

            # (5) Gumbel gate mask (frame-wise)
            gate_logits = self.gumbel_gate(latent_seq)  # [B, T, 1]
            gate_mask = gumbel_sigmoid(gate_logits, tau=0.8)  # [B, T, 1]
            latent_masked = latent_seq * gate_mask  # [B, T, D]

            # (6) Reduce over time (mean with gate weight)
            gate_sum = gate_mask.sum(dim=1) + 1e-6
            z_stage = (latent_masked.sum(dim=1) / gate_sum).contiguous()  # [B, D]

            # (7) Compute mu/logvar per stage
            mu = z_stage.unsqueeze(-1).repeat(1, 1, T)  # for structure alignment
            logvar = torch.zeros_like(mu)  # 可改为 learnable 层

            mu_all.append(mu)
            logvar_all.append(logvar)

            # print(f"[Stage {stage_idx}] mu shape: {mu.shape}, mean: {mu.mean():.4f}, std: {mu.std():.4f}")
            # print(f"[Stage {stage_idx}] logvar shape: {logvar.shape}, mean: {logvar.mean():.4f}, std: {logvar.std():.4f}")

        # (8) 合并多个 stage latent，拼接后做结构分路
        z_concat = torch.cat([m[:, :, 0] for m in mu_all], dim=1)  # [B, D * num_stage]
        z_L, z_R = torch.chunk(z_concat, chunks=2, dim=1)  # shape: [B, 512]

        # (9) 输出左右手 latent 表征的 mu/logvar
        mu_L = self.left_mu(z_L)
        logvar_L = self.left_logvar(z_L)
        mu_R = self.right_mu(z_R)
        logvar_R = self.right_logvar(z_R)

        # (10) 重参数采样
        z_L_sampled = sample_latent(mu_L, logvar_L)
        z_R_sampled = sample_latent(mu_R, logvar_R)

        return z_L_sampled, z_R_sampled, mu_all, logvar_all, mu_L, logvar_L, mu_R, logvar_R
