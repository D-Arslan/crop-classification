"""
Tests — MCTNet complet
Lancer : python -m tests.test_mctnet
         (depuis le dossier "Point 5 — Model Implementation")
"""

import torch
from src.mctnet import MCTNet

torch.manual_seed(42)
B = 4

def test_arkansas():
    """Arkansas : 5 classes — sortie (B, 5)"""
    x    = torch.randn(B, 10, 36)
    mask = (torch.rand(B, 36) > 0.2).float()
    model = MCTNet(n_classes=5)
    logits = model(x, mask)
    assert logits.shape == (B, 5), f"Attendu {(B, 5)}, obtenu {logits.shape}"
    print(f"[OK] Arkansas  — Input {tuple(x.shape)} -> Logits {tuple(logits.shape)}")

def test_california():
    """Californie : 6 classes — sortie (B, 6)"""
    x    = torch.randn(B, 10, 36)
    mask = (torch.rand(B, 36) > 0.2).float()
    model = MCTNet(n_classes=6)
    logits = model(x, mask)
    assert logits.shape == (B, 6), f"Attendu {(B, 6)}, obtenu {logits.shape}"
    print(f"[OK] California — Input {tuple(x.shape)} -> Logits {tuple(logits.shape)}")

def test_shapes_intermediaires():
    """Vérifie les shapes après chaque stage."""
    x    = torch.randn(B, 10, 36)
    mask = (torch.rand(B, 36) > 0.2).float()
    model = MCTNet(n_classes=5)

    out1 = model.stage1(x, mask)
    out2 = model.stage2(out1)
    out3 = model.stage3(out2)
    gmp  = out3.max(dim=2).values

    assert out1.shape == (B, 20, 18), f"Stage 1 : {out1.shape}"
    assert out2.shape == (B, 40,  9), f"Stage 2 : {out2.shape}"
    assert out3.shape == (B, 80,  4), f"Stage 3 : {out3.shape}"
    assert gmp.shape  == (B, 80),     f"GMP     : {gmp.shape}"

    print(f"[OK] Shapes intermediaires :")
    print(f"     Stage 1 : {tuple(out1.shape)}")
    print(f"     Stage 2 : {tuple(out2.shape)}")
    print(f"     Stage 3 : {tuple(out3.shape)}")
    print(f"     GMP     : {tuple(gmp.shape)}")

def test_predict():
    """predict() retourne des indices de classe valides."""
    x    = torch.randn(B, 10, 36)
    mask = (torch.rand(B, 36) > 0.2).float()
    model = MCTNet(n_classes=5)
    preds = model.predict(x, mask)
    assert preds.shape == (B,), f"Attendu ({B},), obtenu {preds.shape}"
    assert preds.min() >= 0 and preds.max() <= 4, f"Classes hors [0,4] : {preds}"
    print(f"[OK] predict() -> {preds.tolist()}  (classes dans [0, 4])")

def test_nombre_parametres():
    """
    Sanity check : l'article annonce 55 059 params pour Arkansas (Table 6).
    On vérifie que notre total est dans le bon ordre de grandeur.
    """
    model_ark = MCTNet(n_classes=5)
    model_cal = MCTNet(n_classes=6)
    total_ark = sum(p.numel() for p in model_ark.parameters())
    total_cal = sum(p.numel() for p in model_cal.parameters())
    print(f"[OK] Parametres Arkansas   : {total_ark:,}  (article Table 6 : 55 059)")
    print(f"[OK] Parametres Californie : {total_cal:,}")

    assert abs(total_ark - 55059) < 10000, (
        f"Trop loin de la cible article (55 059) : {total_ark:,} params. "
        f"Verifier l'architecture."
    )

if __name__ == "__main__":
    print("=" * 60)
    print("Tests — MCTNet complet")
    print("=" * 60)
    test_arkansas()
    test_california()
    test_shapes_intermediaires()
    test_predict()
    test_nombre_parametres()
    print("\nTous les tests sont passes.")
