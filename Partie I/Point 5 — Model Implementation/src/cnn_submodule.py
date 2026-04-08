"""
CNN sub-module — MCTNet
Wang et al., 2024 — "A lightweight CNN-Transformer network for pixel-based
crop mapping using time-series Sentinel-2 imagery"

Implémenté par : Tesnime
Rôle dans le projet : CNN sub-module (Point 5b)

Architecture de référence :
  - Section 2.2 : CNN sub-module (ResBlock 1D)
  - Table 3    : hyperparamètres (kernel_size=3)

Entrée/sortie par CTFusion stage :
  Stage 1 : (B, 10, 36)  →  (B, 10, 36)
  Stage 2 : (B, 20, 18)  →  (B, 20, 18)
  Stage 3 : (B, 40,  9)  →  (B, 40,  9)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# CNN sub-module — Residual Block 1D
# ---------------------------------------------------------------------------

class CNNSubModule(nn.Module):
    """
    CNN sub-module du CTFusion stage (MCTNet).

    Architecture : ResBlock 1D
      Conv1D → BN → Conv1D → BN → ReLU(out + résidu)

    Le résidu permet au gradient de remonter directement à travers les couches,
    évitant le gradient vanishing dans les blocs profonds.

    Args:
        in_channels : C — nombre de canaux d'entrée (et de sortie, inchangé)
        kernel_size : taille du kernel Conv1D (défaut=3, Table 3 de l'article)

    Entrée/sortie : (B, C, T)
      B = taille du batch   (pixels traités simultanément)
      C = nombre de canaux  (bandes spectrales, ou features après fusion)
      T = nombre de timesteps (semaines de la série temporelle)
    """

    def __init__(self, in_channels: int, kernel_size: int = 3):
        super().__init__()

        padding = kernel_size // 2  # padding = 1 pour kernel = 3 → conserve T

        self.conv1 = nn.Conv1d(
            in_channels,    # canaux en entrée  (ex : 10)
            in_channels,    # canaux en sortie  (ex : 10) — dimension C inchangée
            kernel_size=kernel_size,
            padding=padding,
            bias=False,     # inutile car BatchNorm normalise le biais
        )
        self.bn1 = nn.BatchNorm1d(in_channels)

        self.conv2 = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(in_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (B, C, T)

        Returns:
            out : (B, C, T)  — même forme que l'entrée
        """
        residual = x                        # sauvegarde pour la connexion résiduelle

        out = self.conv1(x)                 # (B, C, T)
        out = self.bn1(out)

        out = self.conv2(out)               # (B, C, T)
        out = self.bn2(out)

        out = self.relu(out + residual)     # connexion résiduelle + activation

        return out
