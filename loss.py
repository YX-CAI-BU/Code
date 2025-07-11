import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureLatentLoss(nn.Module):
    def __init__(self, alpha_kl=1.0, beta_var=1e-2, gamma_align=1.0, kl_eps=1e-8):
        super().__init__()
        self.alpha_kl = alpha_kl
        self.beta_var = beta_var
        self.gamma_align = gamma_align
        self.kl_eps = kl_eps

    def forward(self, mu_all, logvar_all):
        """
        mu_all, logvar_all: List of Tensors, each with shape [B, D, T]
        Return: total structure loss (scalar), and individual components as dict
        """
        L_kl_temporal = self.temporal_kl_smoothness(mu_all[-1], logvar_all[-1])
        L_var = self.variance_penalty(logvar_all[-1])
        L_align = self.stage_agreement(mu_all, logvar_all)

        total_loss = (
            self.alpha_kl * L_kl_temporal +
            self.beta_var * L_var +
            self.gamma_align * L_align
        )

        return total_loss, {
            'L_kl_temporal': L_kl_temporal.item(),
            'L_var': L_var.item(),
            'L_align': L_align.item()
        }

    def temporal_kl_smoothness(self, mu, logvar):
        """
        mu, logvar: [B, D, T]
        Smoothness KL across adjacent time steps
        """
        mu1, mu2 = mu[:, :, :-1], mu[:, :, 1:]
        logvar1, logvar2 = logvar[:, :, :-1], logvar[:, :, 1:]
        var1, var2 = torch.exp(logvar1), torch.exp(logvar2)

        kl = 0.5 * (
            (logvar2 - logvar1) +
            (var1 + (mu1 - mu2).pow(2)) / (var2 + self.kl_eps) - 1
        )
        return kl.mean()

    def variance_penalty(self, logvar):
        """
        Penalize large variance (prevent unconstrained expansion)
        logvar: [B, D, T]
        """
        var = torch.exp(logvar)
        return var.mean()

    def stage_agreement(self, mu_all, logvar_all):
        """
        Encourage all stages to align with the last (deepest) one
        Each mu[i], logvar[i]: [B, D, T]
        """
        ref_mu, ref_logvar = mu_all[-1], logvar_all[-1]
        loss = 0.0
        for i in range(len(mu_all) - 1):
            mu_i = mu_all[i]
            logvar_i = logvar_all[i]
            var_i = torch.exp(logvar_i)
            var_ref = torch.exp(ref_logvar)

            kl = 0.5 * (
                (ref_logvar - logvar_i) +
                (var_i + (mu_i - ref_mu).pow(2)) / (var_ref + self.kl_eps) - 1
            )
            loss += kl.mean()
        return loss / (len(mu_all) - 1)
