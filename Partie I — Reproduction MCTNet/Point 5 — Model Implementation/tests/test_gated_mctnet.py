"""
Tests — GatedCTFusion + GatedMCTNet
"""

import torch
from src.ctfusion import CTFusion, GatedCTFusion
from src.mctnet import MCTNet, GatedMCTNet

def test_gated_ctfusion_shapes():
    """Vérifie que GatedCTFusion produit les mêmes formes que CTFusion."""
    B = 4

    configs = [
        dict(in_channels=10, seq_len=36, use_alpe=True,  expected_out=(B, 20, 18)),
        dict(in_channels=20, seq_len=18, use_alpe=False, expected_out=(B, 40,  9)),
        dict(in_channels=40, seq_len= 9, use_alpe=False, expected_out=(B, 80,  4)),
    ]

    for cfg in configs:
        C, T = cfg['in_channels'], cfg['seq_len']
        x    = torch.randn(B, C, T)
        mask = torch.ones(B, T) if cfg['use_alpe'] else None

        block = GatedCTFusion(
            in_channels=C, seq_len=T, use_alpe=cfg['use_alpe']
        )
        out = block(x, mask)

        assert out.shape == cfg['expected_out'], (
            f"GatedCTFusion(C={C}, T={T}) : shape {out.shape} ≠ attendu {cfg['expected_out']}"
        )
        print(f"  ✔  GatedCTFusion(C={C:2d}, T={T:2d}) → {tuple(out.shape)}")

def test_gated_ctfusion_vs_ctfusion_params():
    """GatedCTFusion doit avoir plus de params que CTFusion (gates ajoutées)."""
    configs = [
        dict(in_channels=10, seq_len=36, use_alpe=True),
        dict(in_channels=20, seq_len=18, use_alpe=False),
        dict(in_channels=40, seq_len= 9, use_alpe=False),
    ]

    for cfg in configs:
        C, T = cfg['in_channels'], cfg['seq_len']
        base  = CTFusion(in_channels=C, seq_len=T, use_alpe=cfg['use_alpe'])
        gated = GatedCTFusion(in_channels=C, seq_len=T, use_alpe=cfg['use_alpe'])

        n_base  = sum(p.numel() for p in base.parameters())
        n_gated = sum(p.numel() for p in gated.parameters())
        extra   = n_gated - n_base

        expected_extra = 2 * C * C + C
        assert extra == expected_extra, (
            f"C={C} : extra params {extra} ≠ attendu {expected_extra}"
        )
        print(f"  ✔  C={C:2d} : CTFusion={n_base:,} | GatedCTFusion={n_gated:,} (+{extra} params gate)")

def test_gated_mctnet_shapes():
    """Vérifie les formes de sortie de GatedMCTNet pour Arkansas et California."""
    B = 8
    x    = torch.randn(B, 10, 36)
    mask = torch.ones(B, 36)

    for region, n_classes in [('Arkansas', 5), ('California', 6)]:
        model  = GatedMCTNet(n_classes=n_classes)
        logits = model(x, mask)
        assert logits.shape == (B, n_classes), (
            f"{region} : shape {logits.shape} ≠ ({B}, {n_classes})"
        )
        print(f"  ✔  GatedMCTNet {region} : input {tuple(x.shape)} → logits {tuple(logits.shape)}")

def test_gated_mctnet_params():
    """GatedMCTNet doit avoir ~4 270 params de plus que MCTNet."""
    for n_classes in [5, 6]:
        base  = MCTNet(n_classes=n_classes)
        gated = GatedMCTNet(n_classes=n_classes)

        n_base  = sum(p.numel() for p in base.parameters())
        n_gated = sum(p.numel() for p in gated.parameters())
        extra   = n_gated - n_base

        expected_extra = 210 + 820 + 3240
        assert extra == expected_extra, (
            f"n_classes={n_classes} : extra={extra} ≠ attendu {expected_extra}"
        )
        print(
            f"  ✔  n_classes={n_classes} : MCTNet={n_base:,} | GatedMCTNet={n_gated:,} "
            f"(+{extra} params)"
        )

def test_gated_mctnet_predict():
    """predict() doit retourner des indices de classes valides."""
    B, n_classes = 16, 5
    x    = torch.randn(B, 10, 36)
    mask = torch.ones(B, 36)

    model = GatedMCTNet(n_classes=n_classes)
    preds = model.predict(x, mask)

    assert preds.shape == (B,)
    assert preds.min() >= 0 and preds.max() < n_classes
    print(f"  ✔  predict() → shape {tuple(preds.shape)}, classes ∈ [0, {n_classes-1}]")

def test_gate_values():
    """Les valeurs de la gate doivent être dans (0, 1) — vérifie le sigmoid."""
    B, C, T = 4, 10, 36
    x    = torch.randn(B, C, T)
    mask = torch.ones(B, T)

    block = GatedCTFusion(in_channels=C, seq_len=T, use_alpe=True)
    block.eval()

    with torch.no_grad():
        cnn_out = block.cnn(x)
        tr_out  = block.transformer(x, mask)
        context = torch.cat([cnn_out, tr_out], dim=1).mean(dim=2)
        alpha   = torch.sigmoid(block.gate_fc(context))

    assert alpha.shape == (B, C)
    assert alpha.min() > 0 and alpha.max() < 1
    print(f"  ✔  Gate α ∈ ({alpha.min():.3f}, {alpha.max():.3f}) — bien dans (0, 1)")

if __name__ == '__main__':
    print("\n=== GatedCTFusion — formes de sortie ===")
    test_gated_ctfusion_shapes()

    print("\n=== GatedCTFusion vs CTFusion — paramètres ===")
    test_gated_ctfusion_vs_ctfusion_params()

    print("\n=== GatedMCTNet — formes de sortie ===")
    test_gated_mctnet_shapes()

    print("\n=== GatedMCTNet vs MCTNet — paramètres ===")
    test_gated_mctnet_params()

    print("\n=== GatedMCTNet — predict() ===")
    test_gated_mctnet_predict()

    print("\n=== Gate — valeurs sigmoid ===")
    test_gate_values()

    print("\n✅  Tous les tests passent.")
