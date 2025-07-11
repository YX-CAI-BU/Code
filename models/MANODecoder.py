import torch
import torch.nn as nn
from models.mano_layer import loadManoHandModel
from models.hand_common import LANDMARK

class Decoder(nn.Module):
    def __init__(self, mano_model_dir, device='cuda', latent_dim=256):
        super().__init__()
        self.mano = loadManoHandModel(mano_model_dir, device=device)
        self.device = device
        self.pose_dim = 15
        self.shape_dim = 10
        self.rot_dim = 3
        self.trans_dim = 3
        self.param_dim = self.pose_dim + self.shape_dim + self.rot_dim + self.trans_dim

        # Fully connected decoder: z → MANO param [31]
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.param_dim)  # → 31-dim
        )

        self.valid_landmark_indices = [
            lm.value for name, lm in LANDMARK.__members__.items()
            if name != "PALM_CENTER"
        ]

    def decode_z(self, z, is_right: bool):
        B = z.shape[0]
        mano_param = self.fc(z)  # [B, 31]
        pose = mano_param[:, :self.pose_dim]
        shape = mano_param[:, self.pose_dim:self.pose_dim + self.shape_dim]
        rot = mano_param[:, self.pose_dim + self.shape_dim:self.pose_dim + self.shape_dim + self.rot_dim]
        trans = mano_param[:, -self.trans_dim:]
        global_xform = torch.cat([rot, trans], dim=1)

        joints_list = []
        for i in range(B):
            _, joints = self.mano.forward_kinematics(
                shape[i], pose[i], global_xform[i], torch.tensor([is_right]).to(z.device)
            )
            joints_list.append(joints)  # [20, 3]
        return torch.stack(joints_list, dim=0)  # [B, 20, 3]

    def forward(self, z_left, z_right):
        joints_left = self.decode_z(z_left, is_right=False)
        joints_right = self.decode_z(z_right, is_right=True)
        return joints_left.to(self.device), joints_right.to(self.device)

if __name__ == "__main__":
    decoder = Decoder("/home/ycai3/Hand/ycai3/mano_v1_2/models/")
    z_l = torch.randn(4, 256)
    z_r = torch.randn(4, 256)
    j_l, j_r = decoder(z_l, z_r)
    print("Left Joints:", j_l.shape)   # [4, 20, 3]
    print("Right Joints:", j_r.shape)
