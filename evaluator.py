import torch
import numpy as np


class PoseEvaluator:
    def __init__(self, pck_threshold=20.0, auc_max=50.0, auc_step=1.0, flip_axis=0):
        self.pck_threshold = pck_threshold
        self.auc_max = auc_max
        self.auc_step = auc_step
        self.flip_axis = flip_axis

    @staticmethod
    def mpjpe(pred, gt):
        assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}"
        return ((pred - gt).norm(dim=-1).mean() * 1000.0).item()

    @staticmethod
    def pck(pred, gt, threshold=20.0):
        assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}"
        dist = (pred - gt).norm(dim=-1) * 1000.0
        return (dist < threshold).float().mean().item()

    def auc(self, pred, gt):
        thresholds = np.arange(0, self.auc_max + self.auc_step, self.auc_step)
        auc = 0.0
        for t in thresholds:
            auc += self.pck(pred, gt, threshold=t)
        return auc / len(thresholds)

    def symmetry(self, joints_r, joints_l):
        assert joints_r.shape == joints_l.shape, f"Shape mismatch: {joints_r.shape} vs {joints_l.shape}"
        joints_l_flipped = joints_l.clone()
        joints_l_flipped[..., self.flip_axis] *= -1
        return (joints_r - joints_l_flipped).norm(dim=-1).mean().item()

    @staticmethod
    def smoothness(joint_seq):  # [B, T, J, 3]
        vel = joint_seq[:, 1:] - joint_seq[:, :-1]
        return vel.norm(dim=-1).mean().item()

    def evaluate(self, pred_r, pred_l, gt_r, gt_l):
        # 维度修复：如果 GT 是 [J, B, 3]，转置为 [B, J, 3]
        if gt_r.shape != pred_r.shape and gt_r.shape[0] == pred_r.shape[1]:
            gt_r = gt_r.permute(1, 0, 2)
        if gt_l.shape != pred_l.shape and gt_l.shape[0] == pred_l.shape[1]:
            gt_l = gt_l.permute(1, 0, 2)

        # 再次 sanity check
        assert pred_r.shape == gt_r.shape, f"Shape mismatch: {pred_r.shape} vs {gt_r.shape}"
        assert pred_l.shape == gt_l.shape, f"Shape mismatch: {pred_l.shape} vs {gt_l.shape}"

        return {
            "MPJPE_R": self.mpjpe(pred_r, gt_r),
            "MPJPE_L": self.mpjpe(pred_l, gt_l),
            "PCK_R": self.pck(pred_r, gt_r, self.pck_threshold),
            "PCK_L": self.pck(pred_l, gt_l, self.pck_threshold),
            "AUC_R": self.auc(pred_r, gt_r),
            "AUC_L": self.auc(pred_l, gt_l),
            "Symmetry": self.symmetry(pred_r, pred_l),
        }
