# Point 5 — Model Implementation

## Statut : ⏳ En cours (entraînement à lancer)

## Objectif
Implémenter MCTNet en PyTorch, entraîner sur Arkansas et Californie, et reproduire les résultats de la Table 5 de l'article (OA=0.968 / Kappa=0.951 / F1=0.933 pour Arkansas).

## Fichiers produits
- `src/transformer_alpe.py` — ALPE + Transformer sub-module ✅
- `src/cnn_submodule.py` — CNN sub-module ✅
- `src/ctfusion.py` — bloc CTFusion (CNN // Transformer → Concat → MaxPool1d) ✅
- `src/mctnet.py` — assemblage complet MCTNet ✅
- `tests/test_transformer.py` — tests des 3 stages + cas limites ✅
- `tests/test_ctfusion.py` — tests pipeline 3 stages ✅
- `tests/test_mctnet.py` — pipeline complet + sanity check paramètres ✅
- `doc_transformer_alpe.md` — documentation ALPE + Transformer ✅
- `doc_ctfusion.md` — documentation CTFusion ✅
- `doc_mctnet.md` — documentation MCTNet ✅
- `doc_train.md` — documentation train.py ✅
- `train.py` — boucle d'entraînement complète + métriques OA/Kappa/F1 ✅

---

## Architecture MCTNet — vue d'ensemble

```
Input 1 : (B, 10, 36)   <- bandes x timesteps
Input 2 : (B, 36)       <- masque de manquants (0=manquant, 1=valide)

+--------------------------------------------------+
|  CTFusion Stage 1                                |
|  CNN(10->10, T=36) ---+                          |
|                       +-- Concat+MaxPool -> (B, 20, 18)  |
|  Transformer+ALPE ----+                          |
+--------------------------------------------------+
+--------------------------------------------------+
|  CTFusion Stage 2                                |
|  CNN(20->20, T=18) ---+                          |
|                       +-- Concat+MaxPool -> (B, 40, 9)   |
|  Transformer ---------+                          |
+--------------------------------------------------+
+--------------------------------------------------+
|  CTFusion Stage 3                                |
|  CNN(40->40, T=9) ----+                          |
|                       +-- Concat+MaxPool -> (B, 80, 4)   |
|  Transformer ---------+                          |
+--------------------------------------------------+

Global Max Pooling -> (B, 80)
MLP Classifier    -> (B, N_classes)
```

## Hyperparamètres (Table 3 de l'article)
| Paramètre | Valeur |
|-----------|--------|
| n_stage | 3 |
| n_head | 5 |
| kernel_size | 3 |
| FFN dim | 8 × C |
| optimizer | Adam |
| lr | 0.001 |
| batch_size | 32 |
| epochs | 200 |
| dropout | 0.1 |

## Résultats cibles (Table 5)
| Zone | OA | Kappa | F1 |
|------|-----|-------|-----|
| Arkansas | 0.968 | 0.951 | 0.933 |
| Californie | 0.852 | 0.806 | 0.829 |

## Nombre de paramètres
| Région | Notre implémentation | Article (Table 6) | Ecart |
|--------|---------------------|-------------------|-------|
| Arkansas (5 classes) | 56 798 | 55 059 | +1 739 |
| Californie (6 classes) | 56 799 | — | — |

L'écart résiduel (~1 739 params) est probablement dû à de légères différences dans l'ALPE ou les BN — acceptable pour une reproduction.

---

## 5a. ALPE + Transformer sub-module

**Statut : ✅ Terminé**

### Fichier : `src/transformer_alpe.py`

#### `sinusoidal_pe(seq_len, d_model)`
- Encodage positionnel sinusoïdal (Vaswani et al., 2017)
- Sortie : `(T, C)` — appelé une seule fois dans `ALPE.__init__()`, stocké via `register_buffer`

#### `ECA(channels)` — Efficient Channel Attention
- Global Avg Pool → Conv1D (kernel adaptatif `~log2(C)`) → Sigmoid → multiplication canal
- Entrée/sortie : `(B, C, T)`
- Kernel adaptatif : C=10 → k=3, C=20 → k=3, C=40 → k=5

#### `ALPE(channels, seq_len, kernel_size=3)`
- Formule : `ALPE(t) = ECA(Conv1D(PE(t) * mask))`
- PE → masquage → Conv1D + BN + ReLU → ECA → addition résiduelle sur x
- Entrée : `(B, C, T)` + mask `(B, T)` — Sortie : `(B, C, T)`
- Utilisé **uniquement au Stage 1**

#### `TransformerSubModule(channels, seq_len, n_head=5, use_alpe=False, kernel_size=3)`
- [ALPE optionnel] → MHA(batch_first=True) → Add&Norm → FFN(8xC) → Add&Norm
- Entrée/sortie : `(B, C, T)`

### Décisions d'implémentation
| Décision | Raison |
|----------|--------|
| `batch_first=True` dans MultiheadAttention | Cohérence avec le format `(B, C, T)` |
| FFN dim = 8×C | 8×C donne 56 798 params — le plus proche de la cible article 55 059 |
| PE stocké via `register_buffer` | Calculé une fois dans `__init__`, suit automatiquement `.to(device)` |
| BN après Conv1D dans ALPE | Stabilité d'entraînement |
| Addition résiduelle dans ALPE | L'article formule ALPE comme un ajout à x |

### Historique des modifications
| Version | Modification |
|---------|-------------|
| v1 | Implémentation initiale |
| v2 | PE déplacé de `forward()` vers `__init__()` via `register_buffer` (correction perf) |
| v3 | FFN dim corrigée : 4×C → 8×C (rapprochement de la cible 55 059 params) |

---

## 5b. CNN sub-module

**Statut : ✅ Terminé**

### Fichier : `src/cnn_submodule.py`

Architecture :
- 2 couches Conv1D sur la dimension temporelle
- BatchNorm après chaque couche
- ReLU après les 2 couches
- Connexion résiduelle : `output = input + conv(input)`
- Entrée/sortie : `(B, C, T)`

---

## 5c. CTFusion

**Statut : ✅ Terminé**

### Fichier : `src/ctfusion.py`

- CNN(x) // Transformer(x, mask) → `cat([cnn_out, tr_out], dim=1)` → `MaxPool1d(2)`
- Fusion par concaténation (dim=1) : double les canaux, MaxPool1d(2) réduit la longueur temporelle de moitié
- Stages : (B,10,36)→(B,20,18) → (B,40,9) → (B,80,4)

---

## 5d. MCTNet

**Statut : ✅ Terminé**

### Fichier : `src/mctnet.py`

- `stage1(use_alpe=True)`, `stage2`, `stage3`
- `forward(x, mask)` : stage1 → stage2 → stage3 → GMP → Linear(80, N_classes)
- `predict(x, mask)` : retourne `argmax` des logits (no_grad)

---

## 5e. Entraînement

**Statut : ⏳ À lancer**

### Fichier : `train.py`

```bash
# Scale 30m — Arkansas
python train.py --region Arkansas --data_dir ../../data/preprocessed/scale30

# Scale 30m — Californie
python train.py --region California --data_dir ../../data/preprocessed/scale30

# Scale 20m — Arkansas
python train.py --region Arkansas --data_dir ../../data/preprocessed/scale20

# Scale 20m — Californie
python train.py --region California --data_dir ../../data/preprocessed/scale20
```

Sorties :
- `best_Arkansas_scale30.pth` — meilleur modèle selon F1 val
- `best_California_scale30.pth`
- `best_Arkansas_scale20.pth`
- `best_California_scale20.pth`

Métriques surveillées :
- **OA** (Overall Accuracy) — critère principal de l'article
- **Kappa** (Cohen's Kappa) — robustesse au déséquilibre
- **F1 macro** — critère de sauvegarde du meilleur modèle

---

## Lien avec les autres points
- <- **Point 4** : reçoit les tenseurs `(X, mask, y)` depuis `data/preprocessed/scale30/` ou `scale20/`
- <- **Point 1** : toutes les décisions d'architecture s'appuient sur les sections 2.2-2.4 et Table 3 de l'article
