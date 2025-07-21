import torch
import torch.nn as nn
import torch.nn.functional as F

class EpiPositionalEncoding(nn.Module):
    def __init__(self, num_feats=32):
        super().__init__()
        self.num_feats = num_feats

    def forward(self, grid, K, K_inv, T_lr):
        B, T, H, W, _ = grid.shape
        N = H * W
        uv_norm = grid.reshape(B, T, N, 2)
        uv_pix = (uv_norm + 1) * torch.tensor([W, H], device=grid.device) / 2
        ones = torch.ones(B, T, N, 1, device=grid.device)
        uv1 = torch.cat([uv_pix, ones], dim=-1).reshape(B*T, N, 3).transpose(1,2)
        K_inv_bt = K_inv.reshape(B*T,3,3)
        cam = K_inv_bt @ uv1
        T3  = T_lr[:,:,:3,:3].reshape(B*T,3,3)
        Tt  = T_lr[:,:,:3,3:].reshape(B*T,3,1)
        cam_t = T3 @ cam + Tt
        epi = F.normalize(cam_t - cam, dim=1).transpose(1,2)  # [B*T, N, 3]
        # build sinusoidal PE: total dim = 3 * 2 * num_feats
        pe_list = []
        for i in range(self.num_feats):
            div = 10000 ** (2*(i//2)/self.num_feats)
            pe_list.append(torch.sin(epi/div))
            pe_list.append(torch.cos(epi/div))
        pe = torch.cat(pe_list, dim=-1)  # [B*T, N, 3*2*num_feats]
        return pe.reshape(B, T, H, W, -1)  # [B, T, H, W, D_pe]

class CrossViewDeformableAttention(nn.Module):
    def __init__(self, dim, pe_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        # 项目 pe 到 dim 维
        self.pe_proj = nn.Linear(pe_dim, dim)

    def forward(self, lf, rf, pe):
        # lf, rf: [B, T, C, H, W], pe: [B, T, H, W, D_pe]
        B, T, C, H, W = lf.shape
        N = H * W
        # 展平为 [BT, N, C]
        q = lf.reshape(B*T, C, N).transpose(1,2)
        k = rf.reshape(B*T, C, N).transpose(1,2)
        # pe → [BT, N, D_pe]
        pe_flat = pe.reshape(B*T, N, -1)
        pe_mapped = self.pe_proj(pe_flat)  # [BT, N, C]
        # 加 PE
        q_pe = q + pe_mapped
        k_pe = k + pe_mapped
        # attention
        attn_out, _ = self.attn(q_pe, k_pe, k)
        out = self.norm(attn_out + q)  # 残差
        # reshape 回 [B, T, C, H, W]
        return out.transpose(1,2).reshape(B, T, C, H, W)

class MultiScaleCrossViewFusion(nn.Module):
    def __init__(self, stages=4, dim=128, pe_feats=32, heads=4):
        super().__init__()
        self.pe_modules = nn.ModuleList([EpiPositionalEncoding(pe_feats) for _ in range(stages)])
        pe_dim = 3 * 2 * pe_feats  # epipolar channels
        self.attn_modules = nn.ModuleList([CrossViewDeformableAttention(dim, pe_dim, heads) for _ in range(stages)])

    def forward(self, left_feats, right_feats, K, K_inv, T_lr):
        fused = []
        for i, (lf, rf) in enumerate(zip(left_feats, right_feats)):
            # print(f"--- Fusion Stage {i} Input ---")
            # print(f"  left_feats[{i}]: {lf.shape}")
            # print(f"  right_feats[{i}]:{rf.shape}")

            B, T, C, H, W = lf.shape
            # 构造 grid
            grid = F.affine_grid(
                torch.eye(2,3,device=lf.device).unsqueeze(0).repeat(B*T,1,1),
                size=(B*T, C, H, W), align_corners=False
            ).view(B, T, H, W, 2)

            pe = self.pe_modules[i](grid, K, K_inv, T_lr)
            # print(f"  epi_pe[{i}]: {pe.shape}")

            fused_i = self.attn_modules[i](lf, rf, pe)
            # print(f"  fused_feats[{i}]: {fused_i.shape}")
            fused.append(fused_i)
        return fused