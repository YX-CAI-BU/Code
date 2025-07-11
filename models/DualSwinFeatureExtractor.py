import torch
import torch.nn as nn
import timm

class DualSwinFPN(nn.Module):
    def __init__(self, in_chans=1, out_dims=[96, 192, 384, 768]):
        super().__init__()
        self.swin_left = timm.create_model("swin_tiny_patch4_window7_224", requires_grad=False, pretrained=False, in_chans=in_chans, features_only=True)
        self.swin_right = timm.create_model("swin_tiny_patch4_window7_224", requires_grad=False, pretrained=False, in_chans=in_chans, features_only=True)

        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(dim, 128, kernel_size=1) for dim in out_dims
        ])

    def forward(self, left_img, right_img):
        B, T, C, H, W = left_img.shape
        left_img = left_img.view(B * T, C, H, W)
        right_img = right_img.view(B * T, C, H, W)

        left_feats = self.swin_left(left_img)
        right_feats = self.swin_right(right_img)

        # for i, (lf, rf) in enumerate(zip(left_feats, right_feats)):
        #     print(f"[Raw Swin] Layer {i}: left {lf.shape}, right {rf.shape}")

        left_feats = [feat.permute(0, 3, 1, 2) for feat in left_feats]
        right_feats = [feat.permute(0, 3, 1, 2) for feat in right_feats]

        left_feats = [self.fpn_convs[i](feat) for i, feat in enumerate(left_feats)]
        right_feats = [self.fpn_convs[i](feat) for i, feat in enumerate(right_feats)]

        # for i, (lf, rf) in enumerate(zip(left_feats, right_feats)):
        #     print(f"[FPN Align] Layer {i}: left {lf.shape}, right {rf.shape}")

        left_feats = [feat.view(B, T, feat.shape[1], feat.shape[2], feat.shape[3]) for feat in left_feats]
        right_feats = [feat.view(B, T, feat.shape[1], feat.shape[2], feat.shape[3]) for feat in right_feats]

        return left_feats, right_feats