"""
Tests — ALPE + Transformer sub-module
Lancer : python -m tests.test_transformer
         (depuis le dossier "Point 5 — Model Implementation")
"""

import torch
from src.transformer_alpe import TransformerSubModule

torch.manual_seed(42)
B = 4

def test_stage1_avec_alpe():
    C, T = 10, 36
    x = torch.randn(B, C, T)
    mask = (torch.rand(B, T) > 0.2).float()
    model = TransformerSubModule(channels=C, seq_len=T, n_head=5, use_alpe=True, kernel_size=3)
    out = model(x, mask)
    assert out.shape == (B, C, T), f"Shape attendue {(B, C, T)}, obtenue {out.shape}"
    print(f"[OK] Stage 1 — Input {x.shape}  Mask {mask.shape}  Output {out.shape}")

def test_stage2_sans_alpe():
    C, T = 20, 18
    x = torch.randn(B, C, T)
    model = TransformerSubModule(channels=C, seq_len=T, n_head=5, use_alpe=False)
    out = model(x)
    assert out.shape == (B, C, T), f"Shape attendue {(B, C, T)}, obtenue {out.shape}"
    print(f"[OK] Stage 2 — Input {x.shape}  Output {out.shape}")

def test_stage3_sans_alpe():
    C, T = 40, 9
    x = torch.randn(B, C, T)
    model = TransformerSubModule(channels=C, seq_len=T, n_head=5, use_alpe=False)
    out = model(x)
    assert out.shape == (B, C, T), f"Shape attendue {(B, C, T)}, obtenue {out.shape}"
    print(f"[OK] Stage 3 — Input {x.shape}  Output {out.shape}")

def test_nombre_parametres():
    s1 = TransformerSubModule(channels=10, seq_len=36, n_head=5, use_alpe=True)
    s2 = TransformerSubModule(channels=20, seq_len=18, n_head=5, use_alpe=False)
    s3 = TransformerSubModule(channels=40, seq_len=9,  n_head=5, use_alpe=False)
    total = (sum(p.numel() for p in s1.parameters())
           + sum(p.numel() for p in s2.parameters())
           + sum(p.numel() for p in s3.parameters()))
    print(f"[OK] Nombre total de paramètres (3 Transformers) : {total:,}")

def test_masque_requis_si_alpe():
    """use_alpe=True sans masque doit lever une ValueError."""
    model = TransformerSubModule(channels=10, seq_len=36, n_head=5, use_alpe=True)
    x = torch.randn(B, 10, 36)
    try:
        model(x)
        assert False, "Aurait dû lever ValueError"
    except ValueError:
        print("[OK] ValueError levée correctement si masque absent avec use_alpe=True")

if __name__ == "__main__":
    print("=" * 60)
    print("Tests — ALPE + Transformer sub-module")
    print("=" * 60)
    test_stage1_avec_alpe()
    test_stage2_sans_alpe()
    test_stage3_sans_alpe()
    test_nombre_parametres()
    test_masque_requis_si_alpe()
    print("\nTous les tests sont passés.")
