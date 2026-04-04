"""
Tests du CNN sub-module — MCTNet
Vérifie les 3 stages + propriétés du ResBlock.

Usage :
    python -m tests.test_cnn
"""

import torch
from cnn_submodule import CNNSubModule


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_shapes():
    """Vérifie que l'entrée et la sortie ont bien la même forme."""
    configs = [
        (10, 36),   # Stage 1
        (20, 18),   # Stage 2
        (40,  9),   # Stage 3
    ]

    B = 4  # taille de batch de test
    for stage, (C, T) in enumerate(configs, start=1):
        model = CNNSubModule(in_channels=C)
        x     = torch.randn(B, C, T)
        out   = model(x)
        assert out.shape == (B, C, T), (
            f"Stage {stage} — forme incorrecte : attendu {(B, C, T)}, obtenu {out.shape}"
        )
        print(f"Stage {stage} — Input : {x.shape}  Output : {out.shape}  ✅")


def test_residual():
    """Vérifie que la connexion résiduelle fonctionne (grad non nul sur x)."""
    model = CNNSubModule(in_channels=10)
    x     = torch.randn(4, 10, 36, requires_grad=True)
    out   = model(x)
    out.sum().backward()
    assert x.grad is not None, "Gradient non reçu sur l'entrée — connexion résiduelle cassée"
    print("Connexion résiduelle          ✅")


def test_no_channel_change():
    """Vérifie que C reste constant (contrat avec le Transformer sub-module)."""
    for C in [10, 20, 40]:
        model = CNNSubModule(in_channels=C)
        x     = torch.randn(2, C, 12)
        out   = model(x)
        assert out.shape[1] == C, f"C a changé : {x.shape[1]} → {out.shape[1]}"
    print("Dimension C inchangée         ✅")


def test_param_count():
    """Affiche le nombre de paramètres par stage."""
    configs = [(10, "Stage 1"), (20, "Stage 2"), (40, "Stage 3")]
    total   = 0
    for C, label in configs:
        n = count_parameters(CNNSubModule(in_channels=C))
        total += n
        print(f"Paramètres {label} (C={C:2d}) : {n:,}")
    print(f"Total 3 CNN sub-modules       : {total:,}")
    return total


if __name__ == "__main__":
    print("=" * 55)
    print("Tests — CNN sub-module")
    print("=" * 55)
    test_shapes()
    test_residual()
    test_no_channel_change()
    print("-" * 55)
    test_param_count()
    print("=" * 55)
    print("Tous les tests sont passés.")
