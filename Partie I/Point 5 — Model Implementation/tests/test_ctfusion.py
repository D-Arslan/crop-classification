"""
Tests — CTFusion block
Lancer : python -m tests.test_ctfusion
         (depuis le dossier "Point 5 — Model Implementation")
"""

import torch
from src.ctfusion import CTFusion

torch.manual_seed(42)
B = 4


def test_stage1():
    """Stage 1 : (B, 10, 36) + mask -> (B, 20, 18)"""
    C, T = 10, 36
    x    = torch.randn(B, C, T)
    mask = (torch.rand(B, T) > 0.2).float()
    model = CTFusion(in_channels=C, seq_len=T, n_head=5, use_alpe=True)
    out = model(x, mask)
    assert out.shape == (B, 2*C, T//2), f"Attendu {(B, 2*C, T//2)}, obtenu {out.shape}"
    print(f"[OK] Stage 1 — ({B},{C},{T}) + mask -> {tuple(out.shape)}")


def test_stage2():
    """Stage 2 : (B, 20, 18) -> (B, 40, 9)"""
    C, T = 20, 18
    x = torch.randn(B, C, T)
    model = CTFusion(in_channels=C, seq_len=T, n_head=5, use_alpe=False)
    out = model(x)
    assert out.shape == (B, 2*C, T//2), f"Attendu {(B, 2*C, T//2)}, obtenu {out.shape}"
    print(f"[OK] Stage 2 — ({B},{C},{T}) -> {tuple(out.shape)}")


def test_stage3():
    """Stage 3 : (B, 40, 9) -> (B, 80, 4)"""
    C, T = 40, 9
    x = torch.randn(B, C, T)
    model = CTFusion(in_channels=C, seq_len=T, n_head=5, use_alpe=False)
    out = model(x)
    assert out.shape == (B, 2*C, T//2), f"Attendu {(B, 2*C, T//2)}, obtenu {out.shape}"
    print(f"[OK] Stage 3 — ({B},{C},{T}) -> {tuple(out.shape)}")


def test_pipeline_3_stages():
    """Pipeline complet : (B,10,36) -> (B,20,18) -> (B,40,9) -> (B,80,4)"""
    x    = torch.randn(B, 10, 36)
    mask = (torch.rand(B, 36) > 0.2).float()

    stage1 = CTFusion(in_channels=10, seq_len=36, n_head=5, use_alpe=True)
    stage2 = CTFusion(in_channels=20, seq_len=18, n_head=5, use_alpe=False)
    stage3 = CTFusion(in_channels=40, seq_len=9,  n_head=5, use_alpe=False)

    out1 = stage1(x, mask)   # (B, 20, 18)
    out2 = stage2(out1)      # (B, 40,  9)
    out3 = stage3(out2)      # (B, 80,  4)

    assert out1.shape == (B, 20, 18)
    assert out2.shape == (B, 40,  9)
    assert out3.shape == (B, 80,  4)
    print(f"[OK] Pipeline 3 stages : {tuple(x.shape)} -> {tuple(out1.shape)} -> {tuple(out2.shape)} -> {tuple(out3.shape)}")
    print(f"     -> Global Max Pooling (dans MCTNet) : {tuple(out3.shape)} -> ({B}, 80)")


def test_nombre_parametres():
    stage1 = CTFusion(in_channels=10, seq_len=36, n_head=5, use_alpe=True)
    stage2 = CTFusion(in_channels=20, seq_len=18, n_head=5, use_alpe=False)
    stage3 = CTFusion(in_channels=40, seq_len=9,  n_head=5, use_alpe=False)
    total = (sum(p.numel() for p in stage1.parameters())
           + sum(p.numel() for p in stage2.parameters())
           + sum(p.numel() for p in stage3.parameters()))
    print(f"[OK] Nombre total de paramètres (3 CTFusion) : {total:,}")


if __name__ == "__main__":
    print("=" * 60)
    print("Tests — CTFusion block")
    print("=" * 60)
    test_stage1()
    test_stage2()
    test_stage3()
    test_pipeline_3_stages()
    test_nombre_parametres()
    print("\nTous les tests sont passés.")
