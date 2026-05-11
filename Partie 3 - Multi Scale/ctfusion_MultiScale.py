"""
CTFusion — bloc CNN-Transformer Fusion
MCTNet — Wang et al., 2024

Architecture de référence :
  - Section 2.2 : CTFusion block
  - Figure 3    : CNN // Transformer → Concat → MaxPool

Entrée/sortie par stage :
  Stage 1 : (B, 10, 36) → (B, 20, 18)
  Stage 2 : (B, 20, 18) → (B, 40,  9)
  Stage 3 : (B, 40,  9) → (B, 80,  4)

Note : le Global Max Pooling (B, 80, 4) → (B, 80) est dans mctnet.py, pas ici.
"""

import torch
import torch.nn as nn

from src.transformer_alpe import TransformerSubModule

from src.cnn_submodule import MSCNNSubModule



class CTFusion(nn.Module):
    """
    Bloc CTFusion — unité de base répétée 3 fois dans MCTNet.

    Flux :
      x ──→ CNN        ──→ cnn_out  : (B, C, T)
      x ──→ Transformer ──→ tr_out  : (B, C, T)
      Concat([cnn_out, tr_out], dim=1)  →  (B, 2C, T)
      MaxPool1d(kernel_size=2, stride=2) →  (B, 2C, T//2)

    Le CNN et le Transformer traitent x en parallèle — ils voient
    la même entrée et leurs sorties sont concaténées sur la dimension
    des canaux avant la réduction temporelle par MaxPool.

    Args:
        in_channels  : C — canaux en entrée (10, 20, ou 40 selon le stage)
        seq_len      : T — timesteps en entrée (36, 18, ou 9)
        n_head       : nombre de têtes d'attention (défaut=5, Table 3)
        kernel_size  : kernel Conv1D du CNN et de l'ALPE (défaut=3, Table 3)
        use_alpe     : True uniquement pour le Stage 1
        dropout      : taux de dropout dans le Transformer (défaut=0.1)

    Entrée  : (B, C, T)  +  mask (B, T) si use_alpe=True
    Sortie  : (B, 2C, T//2)
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        n_head: int = 5,
        kernel_size: int = 3,
        use_alpe: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cnn = MSCNNSubModule(in_channels=in_channels)

        self.transformer = TransformerSubModule(
            channels=in_channels,
            seq_len=seq_len,
            n_head=n_head,
            use_alpe=use_alpe,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # MaxPool1d — conforme Figure 3 de l'article
        # kernel_size=2, stride=2 : divise T par 2
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x    : (B, C, T)
            mask : (B, T)  — requis si use_alpe=True (Stage 1 uniquement)

        Returns:
            out  : (B, 2C, T//2)
        """
        # CNN et Transformer en parallèle sur la même entrée x
        cnn_out = self.cnn(x)                    # (B, C, T)
        tr_out  = self.transformer(x, mask)      # (B, C, T)

        # Concaténation sur la dimension canaux
        fused = torch.cat([cnn_out, tr_out], dim=1)   # (B, 2C, T)

        # Réduction temporelle — MaxPool conforme Figure 3
        out = self.pool(fused)                   # (B, 2C, T//2)

        return out
