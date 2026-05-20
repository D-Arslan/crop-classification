"""
CTFusion — bloc CNN-Transformer Fusion
GatedCTFusion — variante avec gate apprise (Partie 3)
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

from src.cnn_submodule import CNNSubModule
from src.transformer_alpe import TransformerSubModule

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

        self.cnn = CNNSubModule(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )

        self.transformer = TransformerSubModule(
            channels=in_channels,
            seq_len=seq_len,
            n_head=n_head,
            use_alpe=use_alpe,
            kernel_size=kernel_size,
            dropout=dropout,
        )

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

        cnn_out = self.cnn(x)
        tr_out  = self.transformer(x, mask)

        fused = torch.cat([cnn_out, tr_out], dim=1)

        out = self.pool(fused)

        return out

class GatedCTFusion(nn.Module):
    """
    GatedCTFusion — variante de CTFusion avec fusion pondérée apprise.

    Problème de CTFusion original : CNN et Transformer contribuent toujours
    à 50/50 via la concaténation fixe. Or leur utilité relative varie selon
    la culture (pics courts → CNN, tendances longues → Transformer) et le stage.

    Solution : une gate apprend un vecteur α ∈ (0,1) par canal à partir du
    contexte global (moyenne temporelle des deux sorties). α pondère la
    contribution du CNN, (1-α) celle du Transformer.

    Flux :
      x ──→ CNN         ──→ cnn_out : (B, C, T)  ──┐
      x ──→ Transformer ──→ tr_out  : (B, C, T)  ──┤
                                                     ├── context (B, 2C)
                                                     ├── gate_fc → α (B, C)
      fused = cat([α·cnn_out, (1-α)·tr_out])  →  (B, 2C, T)
      MaxPool1d(kernel=2, stride=2)            →  (B, 2C, T//2)

    Forme de sortie identique à CTFusion — compatible avec MCTNet sans
    aucune modification du reste du code.

    Params supplémentaires par rapport à CTFusion :
      gate_fc : Linear(2C, C)  →  2C² + C params par stage
      Stage 1 (C=10) :  210 params
      Stage 2 (C=20) :  820 params
      Stage 3 (C=40) : 3 240 params
      Total           : ~4 270 params supplémentaires

    Args:
        in_channels  : C — canaux en entrée
        seq_len      : T — timesteps en entrée
        n_head       : têtes d'attention (défaut=5)
        kernel_size  : kernel Conv1D (défaut=3)
        use_alpe     : True uniquement pour le Stage 1
        dropout      : dropout Transformer (défaut=0.1)

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

        self.cnn = CNNSubModule(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )

        self.transformer = TransformerSubModule(
            channels=in_channels,
            seq_len=seq_len,
            n_head=n_head,
            use_alpe=use_alpe,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.gate_fc = nn.Linear(2 * in_channels, in_channels)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x    : (B, C, T)
            mask : (B, T)  — requis si use_alpe=True

        Returns:
            out  : (B, 2C, T//2)
        """
        cnn_out = self.cnn(x)
        tr_out  = self.transformer(x, mask)

        context = torch.cat([cnn_out, tr_out], dim=1).mean(dim=2)

        alpha = torch.sigmoid(self.gate_fc(context))
        alpha = alpha.unsqueeze(-1)

        fused = torch.cat(
            [alpha * cnn_out, (1 - alpha) * tr_out], dim=1
        )

        return self.pool(fused)
