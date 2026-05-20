
import torch
import torch.nn as nn

class MSCNNSubModule(nn.Module):
    """
    CNN multi-échelle — remplace CNNSubModule dans CTFusion.

    3 branches ResBlock 1D en parallèle avec des kernels différents :
      - kernel 3  → ~30 jours  (variations mensuelles)
      - kernel 7  → ~70 jours  (cycles trimestriels)
      - kernel 15 → ~150 jours (cycle plantation→récolte)

    Les sorties sont concaténées sur la dimension canaux (3C),
    puis ramenées à C par une Conv 1×1 pour que le reste du
    modèle ne change pas du tout.

    Entrée/sortie : (B, C, T)  — forme identique à CNNSubModule
    """

    def __init__(self, in_channels: int):
        super().__init__()

        self.branch3  = self._make_branch(in_channels, kernel_size=3)
        self.branch7  = self._make_branch(in_channels, kernel_size=7)
        self.branch15 = self._make_branch(in_channels, kernel_size=15)

        self.proj = nn.Sequential(
            nn.Conv1d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
        )

    def _make_branch(self, channels: int, kernel_size: int) -> nn.Sequential:
        """Construit un ResBlock 1D pour un kernel donné."""
        pad = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (B, C, T)
        Returns:
            out : (B, C, T)
        """

        out3  = torch.relu(self.branch3(x)  + x)
        out7  = torch.relu(self.branch7(x)  + x)
        out15 = torch.relu(self.branch15(x) + x)

        fused = torch.cat([out3, out7, out15], dim=1)

        return self.proj(fused)
