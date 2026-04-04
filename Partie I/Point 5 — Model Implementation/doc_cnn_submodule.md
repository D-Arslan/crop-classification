# Documentation — CNN sub-module
## `cnn_submodule.py`

---

## Table des matières

1. [Contexte et rôle dans MCTNet](#1-contexte-et-rôle-dans-mctnet)
2. [Vue d'ensemble du fichier](#2-vue-densemble-du-fichier)
3. [CNNSubModule — ResBlock 1D](#3-cnnsubmodule--resblock-1d)
4. [Comment utiliser ce module](#4-comment-utiliser-ce-module)
5. [Ce qui reste à faire](#5-ce-qui-reste-à-faire)

---

## 1. Contexte et rôle dans MCTNet

### Le problème que ce module résout

Dans une série temporelle Sentinel-2, des patterns **locaux** dans le temps sont très informatifs : une montée rapide du NDVI sur 2-3 semaines signale une levée de culture, une chute brusque indique une récolte. Un Transformer seul calcule des relations globales entre tous les timesteps mais n'est pas optimisé pour ces patterns courts et locaux.

Le CNN sub-module résout ce problème en appliquant un filtre glissant (Conv1D, kernel=3) qui regarde **3 timesteps consécutifs** à la fois, capturant ces transitions rapides. Combiné au Transformer (dépendances globales), il forme le cœur du bloc CTFusion.

### Place dans l'architecture globale

```
Input : (B, 10, 36) + mask (B, 36)
         │
    ┌────┴──────────────────────────────┐
    │  CTFusion Stage 1                 │
    │  CNNSubModule(C=10) ──────┐        │
    │                           ├── Fusion ──→ (B, 20, 18)
    │  Transformer+ALPE(C=10) ──┘        │
    └───────────────────────────────────┘
         │
    ┌────┴──────────────────────────────┐
    │  CTFusion Stage 2                 │
    │  CNNSubModule(C=20) ──────┐        │
    │                           ├── Fusion ──→ (B, 40, 9)
    │  Transformer(C=20) ───────┘        │
    └───────────────────────────────────┘
         │
    ┌────┴──────────────────────────────┐
    │  CTFusion Stage 3                 │
    │  CNNSubModule(C=40) ──────┐        │
    │                           ├── Fusion ──→ (B, 80, 1)
    │  Transformer(C=40) ───────┘        │
    └───────────────────────────────────┘
         │
    Global Max Pooling → (B, 80)
    MLP Classifier    → (B, N_classes)
```

**Règle importante** : le `CNNSubModule` **ne modifie pas la forme** du tenseur. Il reçoit `(B, C, T)` et produit `(B, C, T)` — c'est le futur bloc CTFusion qui doublera les canaux et divisera T.

---

## 2. Vue d'ensemble du fichier

```
cnn_submodule.py
│
└── class CNNSubModule     ← ResBlock 1D (conv → bn → conv → bn → relu + résidu)
```

### Convention de format

Même convention que `transformer_alpe.py` :
- `B` = taille du batch
- `C` = nombre de canaux
- `T` = nombre de timesteps

---

## 3. `CNNSubModule` — ResBlock 1D

### Architecture

```
x : (B, C, T)
│
├── résidu = x                            ← sauvegarde pour le skip
│
├── Conv1D(C→C, k=3, pad=1, bias=False)
│   BatchNorm1d(C)
│
├── Conv1D(C→C, k=3, pad=1, bias=False)
│   BatchNorm1d(C)
│
└── ReLU( out + résidu )  ──→  (B, C, T)
```

### Conv1D sur la dimension temporelle

`kernel_size=3` avec `padding=1` : le filtre regarde 3 timesteps consécutifs, et le padding conserve la longueur T. Avec `bias=False` car le BatchNorm qui suit normalise de toute façon le biais.

### Connexion résiduelle (skip connection)

```python
out = relu(conv2(conv1(x)) + x)
```

Sans résidu, empiler des couches risque de dégrader le signal (gradient vanishing). Avec le résidu, si les convolutions n'apportent rien, le réseau peut apprendre à les "ignorer" (poids → 0). C'est l'idée des ResNets (He et al., 2016).

Ici le résidu est direct (pas de projection) car `in_channels == out_channels` — la forme ne change pas.

### BatchNorm après chaque Conv

La BatchNorm normalise les activations sur le batch pour chaque canal, stabilisant l'entraînement. Elle est appliquée **avant** la ReLU finale (mais après les convolutions), ce qui est le schéma classique des ResNets.

### Paramètres entraînables

| Couche | Formule | Stage 1 (C=10) | Stage 2 (C=20) | Stage 3 (C=40) |
|--------|---------|----------------|----------------|----------------|
| Conv1D ×1 | C×C×k | 300 | 1 200 | 4 800 |
| BatchNorm1d ×1 | 2×C | 20 | 40 | 80 |
| Conv1D ×2 | C×C×k | 300 | 1 200 | 4 800 |
| BatchNorm1d ×2 | 2×C | 20 | 40 | 80 |
| **Total stage** | | **640** | **2 480** | **9 760** |
| **Total 3 stages** | | | | **12 880** |

---

## 4. Comment utiliser ce module

### Instanciation pour les 3 stages

```python
from cnn_submodule import CNNSubModule

cnn_s1 = CNNSubModule(in_channels=10)  # Stage 1
cnn_s2 = CNNSubModule(in_channels=20)  # Stage 2
cnn_s3 = CNNSubModule(in_channels=40)  # Stage 3
```

### Appel dans le forward

```python
out_s1 = cnn_s1(x)   # x:(B,10,36) → (B,10,36)
out_s2 = cnn_s2(x)   # x:(B,20,18) → (B,20,18)
out_s3 = cnn_s3(x)   # x:(B,40, 9) → (B,40, 9)
```

### Contrat avec le Transformer sub-module (collègue)

Le CNN et le Transformer reçoivent **le même tenseur d'entrée** et produisent tous les deux une sortie de **même forme**. Le bloc CTFusion fusionnera ensuite les deux sorties.

```
x_cnn         = cnn(x)          # (B, C, T)
x_transformer = transformer(x)  # (B, C, T)
# fusion à implémenter dans CTFusion
```

### Test de bon fonctionnement

```bash
# depuis la racine du projet
python -m tests.test_cnn
```

Doit afficher :
```
Stage 1 — Input : torch.Size([4, 10, 36])  Output : torch.Size([4, 10, 36])  ✅
Stage 2 — Input : torch.Size([4, 20, 18])  Output : torch.Size([4, 20, 18])  ✅
Stage 3 — Input : torch.Size([4, 40,  9])  Output : torch.Size([4, 40,  9])  ✅
Connexion résiduelle          ✅
Dimension C inchangée         ✅
--------------------------------------------------
Paramètres Stage 1 (C=10) : 640
Paramètres Stage 2 (C=20) : 2,480
Paramètres Stage 3 (C=40) : 9,760
Total 3 CNN sub-modules       : 12,880
==================================================
Tous les tests sont passés.
```

---

## 5. Ce qui reste à faire

### 🔲 Prochaine étape : bloc CTFusion

Une fois les deux sub-modules disponibles (`cnn_submodule.py` + `transformer_alpe.py`), implémenter `CTFusion` qui :
1. Reçoit `(B, C, T)` + mask `(B, T)`
2. Passe dans `CNNSubModule` → `(B, C, T)`
3. Passe dans `TransformerSubModule` → `(B, C, T)`
4. Fusionne les deux → `(B, 2C, T//2)` (double les canaux, divise le temps par 2)

Le mécanisme exact de fusion (concaténation + Conv ? Addition ? Autre ?) n'est pas précisé dans l'article et devra être décidé en commun.
