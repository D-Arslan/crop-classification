# Point 5 — Model Implementation

## Statut : ⏳ En cours

## Objectif
Implémenter MCTNet en PyTorch, entraîner sur Arkansas et Californie, et reproduire les résultats de la Table 5 de l'article (OA=0.968 / Kappa=0.951 / F1=0.933 pour Arkansas).

## Fichiers produits
- `src/transformer_alpe.py` — ALPE + Transformer sub-module ✅
- `tests/test_transformer.py` — tests des 3 stages + cas limites ✅
- `cnn_submodule.py` — CNN sub-module ✅
- `tests/test_cnn.py` — tests des 3 stages + connexion résiduelle ✅
- `mctnet.py` — assemblage complet MCTNet *(à venir)*
- `train.py` — boucle d'entraînement + évaluation *(à venir)*

---

## Architecture MCTNet — vue d'ensemble

```
Input 1 : (B, 10, 36)   ← bandes × timesteps
Input 2 : (B, 36)       ← masque de manquants (0=manquant, 1=valide)

┌─────────────────────────────────────────────────┐
│  CTFusion Stage 1                               │
│  CNN(10→10, T=36) ──┐                           │
│                     ├── Fusion ── (B, 20, 18)  │
│  Transformer+ALPE ──┘                           │
└─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────┐
│  CTFusion Stage 2                               │
│  CNN(20→20, T=18) ──┐                           │
│                     ├── Fusion ── (B, 40, 9)   │
│  Transformer ───────┘                           │
└─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────┐
│  CTFusion Stage 3                               │
│  CNN(40→40, T=9) ───┐                           │
│                     ├── Fusion ── (B, 80, 1)   │
│  Transformer ───────┘                           │
└─────────────────────────────────────────────────┘

Global Max Pooling → (B, 80)
MLP Classifier    → (B, N_classes)
```

**Note sur la fusion** : le mécanisme exact de fusion CNN+Transformer n'est pas détaillé dans l'article (concatenation ? addition ? attention ?). À clarifier lors de l'implémentation du bloc CTFusion.

## Hyperparamètres (Table 3 de l'article)
| Paramètre | Valeur |
|-----------|--------|
| n_stage | 3 |
| n_head | 5 |
| kernel_size | 3 |
| optimizer | Adam |
| lr | 0.001 |
| batch_size | 32 |
| epochs | 200 |

## Résultats cibles (Table 5)
| Zone | OA | Kappa | F1 |
|------|-----|-------|-----|
| Arkansas | 0.968 | 0.951 | 0.933 |
| Californie | 0.852 | 0.806 | 0.829 |

---

## 5a. ALPE + Transformer sub-module ← Arslan

**Statut : ⏳ Implémentation de base terminée — intégration CTFusion à faire**

### Fichier : `transformer_alpe.py`

### Modules implémentés

#### `sinusoidal_pe(seq_len, d_model)`
- Encodage positionnel sinusoïdal (Vaswani et al., 2017)
- Sortie : `(T, C)`

#### `ECA(channels)` — Efficient Channel Attention
- Global Avg Pool → Conv1D (kernel adaptatif `~log2(C)`) → Sigmoid
- Entrée/sortie : `(B, C, T)`
- Kernel adaptatif : C=10 → k=3, C=20 → k=3, C=40 → k=5

#### `ALPE(channels, seq_len, kernel_size=3)`
- Formule : `ALPE(t) = ECA(Conv1D(PE(t) ⊙ mask))`
- PE → masquage → Conv1D + BN + ReLU → ECA → addition résiduelle
- Entrée : `(B, C, T)` + mask `(B, T)` — Sortie : `(B, C, T)`
- Utilisé **uniquement au Stage 1**

#### `TransformerSubModule(channels, seq_len, n_head=5, use_alpe=False, kernel_size=3)`
- [ALPE optionnel] → MHA → Add&Norm → FFN → Add&Norm
- FFN dim cachée = 4×C (convention standard, non précisée dans l'article)
- Entrée/sortie : `(B, C, T)`

### Tests validés
| Stage | Input | Mask | Output |
|-------|-------|------|--------|
| 1 (avec ALPE) | `(B, 10, 36)` | `(B, 36)` | `(B, 10, 36)` |
| 2 | `(B, 20, 18)` | — | `(B, 20, 18)` |
| 3 | `(B, 40, 9)` | — | `(B, 40, 9)` |

Nombre total de paramètres (3 Transformers) : **26 433**

### Reste à faire
- 🔲 Intégration dans le bloc CTFusion (avec le CNN sub-module)
- 🔲 Implémentation de `mctnet.py` (assemblage complet)
- 🔲 Tests sur données réelles (Arkansas_10k.csv / California_10k.csv)

---

## 5b. CNN sub-module ← Tesnime

**Statut : ✅ Implémentation terminée — intégration CTFusion à faire**

### Fichier : `cnn_submodule.py`

### Module implémenté

#### `CNNSubModule(in_channels, kernel_size=3)` — ResBlock 1D
- Architecture : Conv1D → BN → Conv1D → BN → ReLU(out + résidu)
- Connexion résiduelle directe (pas de projection — C inchangé)
- Entrée/sortie : `(B, C, T)` — forme strictement conservée

### Décisions d'implémentation
| Décision | Raison |
|----------|--------|
| `bias=False` dans Conv1D | Inutile car BatchNorm normalise de toute façon |
| BN après chaque Conv | Stabilité d'entraînement (schéma ResNet classique) |
| Connexion résiduelle directe | `in_channels == out_channels` → pas besoin de projection |
| ReLU après addition résiduelle | Schéma `conv → bn → conv → bn → relu(+résidu)` standard |

### Tests validés
| Stage | Input | Output |
|-------|-------|--------|
| 1 | `(B, 10, 36)` | `(B, 10, 36)` |
| 2 | `(B, 20, 18)` | `(B, 20, 18)` |
| 3 | `(B, 40,  9)` | `(B, 40,  9)` |
| Connexion résiduelle | gradient non nul sur x | ✅ |
| Dimension C inchangée | C identique entrée/sortie | ✅ |

Nombre total de paramètres (3 CNN sub-modules) : **12 880**

### Reste à faire
- 🔲 Intégration dans le bloc CTFusion (avec le Transformer sub-module d'Arslan)
- 🔲 Implémentation de `mctnet.py` (assemblage complet)
- 🔲 Tests sur données réelles (Arkansas / California)

---

## Lien avec les autres points
- ← **Point 4** : reçoit les tenseurs `(X, mask, y)` au format `(B, 10, 36)`, `(B, 36)`, `(B,)`
- ← **Point 1** : toutes les décisions d'architecture s'appuient sur les sections 2.2-2.4 et Table 3 de l'article
