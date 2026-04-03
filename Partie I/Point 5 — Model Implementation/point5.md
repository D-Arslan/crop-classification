# Point 5 — Model Implementation

## Statut : ⏳ En cours

## Objectif
Implémenter MCTNet en PyTorch, entraîner sur Arkansas et Californie, et reproduire les résultats de la Table 5 de l'article (OA=0.968 / Kappa=0.951 / F1=0.933 pour Arkansas).

## Fichiers produits
- `transformer_alpe.py` — ALPE + Transformer sub-module ✅
- `cnn_submodule.py` — CNN sub-module *(Tesnime, à venir)*
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

## 5a. ALPE + Transformer sub-module ← [Moi]

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

### Décisions d'implémentation
| Décision | Raison |
|----------|--------|
| `batch_first=True` dans MultiheadAttention | Cohérence avec le format `(B, C, T)` |
| FFN dim cachée = 4×C | Convention Transformer standard |
| BN après Conv1D dans ALPE | Stabilité d'entraînement |
| Addition résiduelle dans ALPE | L'article formule ALPE comme un ajout à x |

### Tests validés
| Stage | Input | Mask | Output |
|-------|-------|------|--------|
| 1 (avec ALPE) | `(B, 10, 36)` | `(B, 36)` | `(B, 10, 36)` |
| 2 | `(B, 20, 18)` | — | `(B, 20, 18)` |
| 3 | `(B, 40, 9)` | — | `(B, 40, 9)` |

Nombre total de paramètres (3 Transformers) : **26 433**

### Reste à faire
- 🔲 Intégration dans le bloc CTFusion (avec le CNN du coéquipier A)
- 🔲 Implémentation de `mctnet.py` (assemblage complet)
- 🔲 Tests sur données réelles (Arkansas_10k.csv / California_10k.csv)

### Historique des modifications
| Version | Modification |
|---------|-------------|
| v1 | Implémentation initiale |
| v2 | PE sinusoïdal déplacé de `forward()` vers `__init__()` via `register_buffer` — évite le recalcul à chaque passage |

---

## 5b. CNN sub-module ← Coéquipier A

**Statut : ⏳ En cours**

### Architecture attendue
- 2 couches Conv1D sur la dimension temporelle
- BatchNorm après chaque couche
- ReLU après les 2 couches
- Connexion résiduelle : output = input + conv(input)
- Entrée/sortie : `(B, C, T)`

---

## Lien avec les autres points
- ← **Point 4** : reçoit les tenseurs `(X, mask, y)` au format `(B, 10, 36)`, `(B, 36)`, `(B,)`
- ← **Point 1** : toutes les décisions d'architecture s'appuient sur les sections 2.2-2.4 et Table 3 de l'article
