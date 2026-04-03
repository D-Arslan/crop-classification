"""
ALPE + Transformer sub-module — MCTNet
Wang et al., 2024 — "A lightweight CNN-Transformer network for pixel-based
crop mapping using time-series Sentinel-2 imagery"

Implémenté par : Arslan !!
Rôle dans le projet : ALPE + Transformer sub-module (Point 5)

Architecture de référence :
  - Section 2.3 : Transformer sub-module
  - Section 2.4 : ALPE (Adaptive Learnable Positional Encoding)
  - Table 3    : hyperparamètres (n_head=5, kernel_size=3)

Entrée par CTFusion stage :
  Stage 1 : (B, 10, 36)  + mask (B, 36)
  Stage 2 : (B, 20, 18)  — pas d'ALPE
  Stage 3 : (B, 40,  9)  — pas d'ALPE
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilitaire : encodage positionnel sinusoïdal classique
# ---------------------------------------------------------------------------

def sinusoidal_pe(seq_len: int, d_model: int, device=None) -> torch.Tensor:
    """
    Retourne la matrice d'encodage positionnel sinusoïdal.

    Args:
        seq_len : longueur de la séquence temporelle (T)
        d_model : dimension des features (C = nombre de canaux)
        device  : device cible

    Returns:
        pe : Tensor de forme (T, C)
    """
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float, device=device)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    # Pour d_model impair, les indices 1::2 peuvent avoir une dimension < div_term
    pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
    return pe  # (T, C)


# ---------------------------------------------------------------------------
# ECA — Efficient Channel Attention (Wang et al., 2020)
# ---------------------------------------------------------------------------

class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA-Net).
    Applique une attention inter-canaux avec une conv1D locale (kernel adaptatif).

    Args:
        channels : nombre de canaux C
        gamma, b : paramètres pour calculer le kernel adaptatif (défaut article ECA)
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # Taille de kernel adaptative : k = odd(log2(C)/gamma + b/gamma)
        t = int(abs(math.log2(channels) / gamma + b / gamma))
        k = t if t % 2 else t + 1  # force impair
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, T)
        Returns:
            out : (B, C, T)  — x pondéré par l'attention canal
        """
        # Global average pooling sur T → (B, C, 1)
        y = self.avg_pool(x)
        # Conv1D sur la dimension canal (vue comme séquence de longueur C)
        y = y.squeeze(-1).unsqueeze(1)   # (B, 1, C)
        y = self.conv(y)                 # (B, 1, C)
        y = self.sigmoid(y)
        y = y.squeeze(1).unsqueeze(-1)   # (B, C, 1)
        return x * y.expand_as(x)


# ---------------------------------------------------------------------------
# ALPE — Adaptive Learnable Positional Encoding
# ---------------------------------------------------------------------------

class ALPE(nn.Module):
    """
    Adaptive Learnable Positional Encoding (Section 2.4 de l'article).

    ALPE(t) = ECA( Conv1D( PE(t) ⊙ mask ) )

    Étapes :
      1. PE sinusoïdal  : (T, C)
      2. ⊙ mask         : met à 0 les timesteps manquants
      3. Conv1D         : capture les relations temporelles locales
      4. ECA            : pondération inter-canaux
      5. Addition à x   : encodage positionnel résiduel

    Args:
        channels    : C (nombre de canaux / bandes spectrales)
        seq_len     : T (nombre de timesteps)
        kernel_size : taille du kernel Conv1D (défaut=3, Table 3)
    """

    def __init__(self, channels: int, seq_len: int, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len

        # PE pré-calculé une fois, stocké comme buffer (non entraînable,
        # suit automatiquement le device avec .to() / .cuda())
        pe = sinusoidal_pe(seq_len, channels)
        self.register_buffer('pe', pe)  # (T, C)

        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(channels)
        self.eca = ECA(channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : (B, C, T)  — features du CNN (Input 1 réorganisé)
            mask : (B, T)     — 1 = valide, 0 = manquant (Input 2)

        Returns:
            out  : (B, C, T)  — x + encodage positionnel adaptatif
        """
        B = x.shape[0]

        # 1. Encodage positionnel sinusoïdal → (T, C)  [pré-calculé dans __init__]
        pe = self.pe

        # 2. Masquage : met à 0 les positions temporelles manquantes
        #    mask : (B, T) → (B, 1, T) pour broadcast sur C
        mask_bc = mask.unsqueeze(1).float()       # (B, 1, T)
        #    pe : (T, C) → (C, T) → (1, C, T) pour broadcast sur B
        pe_bct = pe.T.unsqueeze(0)                # (1, C, T)
        masked_pe = pe_bct * mask_bc              # (B, C, T)

        # 3. Conv1D sur la dimension temporelle
        out = self.conv(masked_pe)                # (B, C, T)
        out = self.bn(out)
        out = F.relu(out)

        # 4. ECA — attention inter-canaux
        out = self.eca(out)                       # (B, C, T)

        # 5. Addition résiduelle à x
        return x + out


# ---------------------------------------------------------------------------
# Transformer sub-module
# ---------------------------------------------------------------------------

class TransformerSubModule(nn.Module):
    """
    Transformer sub-module du MCTNet (Section 2.3).

    Architecture (encodeur Transformer standard) :
      [Optionnel : ALPE] → Multi-Head Self-Attention → Add & Norm
                        → Feed Forward              → Add & Norm

    Args:
        channels    : C (dimension des features = d_model)
        seq_len     : T (longueur de la séquence)
        n_head      : nombre de têtes d'attention (défaut=5, Table 3)
        use_alpe    : True pour le 1er stage uniquement
        kernel_size : kernel Conv1D de l'ALPE (défaut=3)
        dropout     : taux de dropout (défaut=0.1)
    """

    def __init__(
        self,
        channels: int,
        seq_len: int,
        n_head: int = 5,
        use_alpe: bool = False,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_alpe = use_alpe

        # ALPE — uniquement pour le 1er CTFusion stage
        if use_alpe:
            self.alpe = ALPE(channels, seq_len, kernel_size)

        # Multi-Head Self-Attention
        # PyTorch attend (T, B, C) ou (B, T, C) avec batch_first=True
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(channels)

        # Feed Forward Network (FFN)
        # L'article ne précise pas la dim cachée ; convention Transformer : 4 × d_model
        ffn_dim = 4 * channels
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, channels),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(channels)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x    : (B, C, T)  — sortie du CNN sub-module
            mask : (B, T)     — masque de données manquantes (requis si use_alpe=True)

        Returns:
            out  : (B, C, T)
        """
        B, C, T = x.shape

        # --- ALPE (1er stage seulement) ---
        if self.use_alpe:
            if mask is None:
                raise ValueError("Le masque est requis quand use_alpe=True")
            x = self.alpe(x, mask)          # (B, C, T)

        # Reformat pour MultiheadAttention : (B, T, C)
        x_t = x.permute(0, 2, 1)           # (B, T, C)

        # --- Multi-Head Self-Attention + Add & Norm ---
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x_t = self.norm1(x_t + attn_out)   # Add & Norm

        # --- Feed Forward + Add & Norm ---
        ffn_out = self.ffn(x_t)
        x_t = self.norm2(x_t + ffn_out)    # Add & Norm

        # Retour au format (B, C, T)
        out = x_t.permute(0, 2, 1)
        return out


# ---------------------------------------------------------------------------
# Test rapide (à lancer directement : python transformer_alpe.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    B = 4  # batch size

    print("=" * 60)
    print("Test ALPE + Transformer sub-module — MCTNet")
    print("=" * 60)

    # --- Stage 1 : C=10, T=36 (avec ALPE) ---
    C1, T1 = 10, 36
    x1 = torch.randn(B, C1, T1)
    mask1 = (torch.rand(B, T1) > 0.2).float()  # ~80% valide

    transformer_s1 = TransformerSubModule(
        channels=C1, seq_len=T1, n_head=5, use_alpe=True, kernel_size=3
    )
    out1 = transformer_s1(x1, mask1)
    print(f"Stage 1 — Input : {x1.shape}  Mask : {mask1.shape}  Output : {out1.shape}")
    assert out1.shape == (B, C1, T1), "Erreur shape stage 1"

    # --- Stage 2 : C=20, T=18 (sans ALPE) ---
    C2, T2 = 20, 18
    x2 = torch.randn(B, C2, T2)

    transformer_s2 = TransformerSubModule(
        channels=C2, seq_len=T2, n_head=5, use_alpe=False
    )
    out2 = transformer_s2(x2)
    print(f"Stage 2 — Input : {x2.shape}  Output : {out2.shape}")
    assert out2.shape == (B, C2, T2), "Erreur shape stage 2"

    # --- Stage 3 : C=40, T=9 (sans ALPE) ---
    C3, T3 = 40, 9
    x3 = torch.randn(B, C3, T3)

    transformer_s3 = TransformerSubModule(
        channels=C3, seq_len=T3, n_head=5, use_alpe=False
    )
    out3 = transformer_s3(x3)
    print(f"Stage 3 — Input : {x3.shape}  Output : {out3.shape}")
    assert out3.shape == (B, C3, T3), "Erreur shape stage 3"

    # --- Compter les paramètres ---
    total = sum(p.numel() for p in transformer_s1.parameters())
    total += sum(p.numel() for p in transformer_s2.parameters())
    total += sum(p.numel() for p in transformer_s3.parameters())
    print(f"\nNombre total de paramètres (3 Transformers) : {total:,}")

    print("\nTous les tests sont passés.")
