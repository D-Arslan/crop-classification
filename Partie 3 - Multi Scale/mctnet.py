"""
MCTNet — Multi-stage CNN-Transformer Network
Wang et al., 2024 — "A lightweight CNN-Transformer network for pixel-based
crop mapping using time-series Sentinel-2 imagery"

Architecture de référence :
  - Section 2.2 : CTFusion blocks (x3)
  - Section 2.3.1 : Global Max Pooling
  - Section 2.3.2 : MLP Classifier (Linear + Softmax)
  - Table 6 : 55 059 paramètres pour Arkansas (sanity check)

Flux complet :
  (B, 10, 36) + mask (B, 36)
  -> CTFusion x3
  -> Global Max Pooling
  -> Linear(80 -> N_classes)
  -> (B, N_classes)  [logits]
"""

import torch
import torch.nn as nn

from src.ctfusion import CTFusion


class MCTNet(nn.Module):
    """
    MCTNet — Multi-stage CNN-Transformer Network.

    Args:
        n_classes   : nombre de classes (5 pour Arkansas, 6 pour Californie)
        n_head      : nombre de têtes d'attention (défaut=5, Table 3)
        kernel_size : kernel Conv1D CNN et ALPE (défaut=3, Table 3)
        dropout     : dropout dans le Transformer (défaut=0.1)

    Entrée :
        x    : (B, 10, 36)  — features normalisées (Input 1)
        mask : (B, 36)      — masque de manquants  (Input 2)

    Sortie :
        logits : (B, N_classes)  — pas de softmax (intégré dans CrossEntropyLoss)
    """

    def __init__(
        self,
        n_classes: int,
        n_head: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 3 CTFusion stages
        # Stage 1 : (B, 10, 36) -> (B, 20, 18)  — avec ALPE
        self.stage1 = CTFusion(
            in_channels=10, seq_len=36,
            n_head=n_head, kernel_size=kernel_size,
            use_alpe=True, dropout=dropout,
        )
        # Stage 2 : (B, 20, 18) -> (B, 40, 9)
        self.stage2 = CTFusion(
            in_channels=20, seq_len=18,
            n_head=n_head, kernel_size=kernel_size,
            use_alpe=False, dropout=dropout,
        )
        # Stage 3 : (B, 40, 9) -> (B, 80, 4)
        self.stage3 = CTFusion(
            in_channels=40, seq_len=9,
            n_head=n_head, kernel_size=kernel_size,
            use_alpe=False, dropout=dropout,
        )

        # MLP Classifier (Section 2.3.2)
        # "a linear layer with a Softmax activation"
        # Softmax absent ici — intégré dans CrossEntropyLoss en training
        self.classifier = nn.Linear(80, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : (B, 10, 36)
            mask : (B, 36)

        Returns:
            logits : (B, N_classes)
        """
        # 3 CTFusion stages
        out = self.stage1(x, mask)   # (B, 20, 18)
        out = self.stage2(out)       # (B, 40,  9)
        out = self.stage3(out)       # (B, 80,  4)

        # Global Max Pooling : (B, 80, 4) -> (B, 80)
        out = out.max(dim=2).values  # max sur la dimension temporelle

        # MLP Classifier : (B, 80) -> (B, N_classes)
        logits = self.classifier(out)

        return logits

    def predict(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Inférence — retourne les classes prédites (pas les logits).

        Args:
            x    : (B, 10, 36)
            mask : (B, 36)

        Returns:
            preds : (B,)  — indices des classes prédites
        """
        with torch.no_grad():
            logits = self.forward(x, mask)
            return logits.argmax(dim=1)
