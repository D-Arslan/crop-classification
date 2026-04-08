# Documentation — MCTNet complet
## `src/mctnet.py`

---

## Table des matières

1. [Contexte et rôle](#1-contexte-et-rôle)
2. [Architecture complète](#2-architecture-complète)
3. [Décisions d'implémentation](#3-décisions-dimplémentation)
4. [Problèmes rencontrés](#4-problèmes-rencontrés)
5. [Comment utiliser ce module](#5-comment-utiliser-ce-module)
6. [Ce qui reste à faire](#6-ce-qui-reste-à-faire)

---

## 1. Contexte et rôle

`MCTNet` est le module racine — il assemble tous les sous-modules en un seul
réseau utilisable directement pour l'entraînement et l'inférence.
Il reçoit les tenseurs produits par le preprocessing de Sarah et retourne
les logits de classification.

---

## 2. Architecture complète

```
x : (B, 10, 36)  +  mask : (B, 36)
        |
    CTFusion stage1 (use_alpe=True)    (B, 10, 36) -> (B, 20, 18)
        |
    CTFusion stage2                    (B, 20, 18) -> (B, 40,  9)
        |
    CTFusion stage3                    (B, 40,  9) -> (B, 80,  4)
        |
    Global Max Pooling                 (B, 80,  4) -> (B, 80)
    out.max(dim=2).values
        |
    Linear(80 -> N_classes)            (B, 80)     -> (B, N_classes)
        |
    logits : (B, N_classes)
```

---

## 3. Décisions d'implémentation

### MLP à une seule couche — conforme article
Section 2.3.2 : *"The MLP classifier is made up of a linear layer with a
Softmax activation"*. Nous implémentons `Linear(80 → N_classes)` sans couche
cachée, fidèle à cette description.

La Softmax n'est pas incluse dans `forward()` — `CrossEntropyLoss` de PyTorch
l'intègre nativement (elle applique LogSoftmax + NLLLoss). En inférence,
utiliser `predict()` qui appelle `argmax` directement sur les logits.

Une alternative à deux couches `Linear(80→80) → ReLU → Linear(80→N)` est
documentée dans `idees_partie3.md` comme piste d'amélioration.

### N_classes en paramètre
Arkansas = 5 classes, Californie = 6 classes. `n_classes` est un argument
du constructeur pour que le même code serve les deux datasets.

### FFN dim cachée = 8×C — retrouvée par sanity check
L'article (Table 6) annonce 55 059 paramètres pour Arkansas. Notre implémentation
avec `ffn_dim = 8×C` donne 56 798 params — écart de 1 739 (3%).

| ffn_dim | Params Arkansas | Ecart article |
|---------|----------------|---------------|
| 4×C | 39 718 | -15 341 |
| 6×C | 48 258 | -6 801 |
| **8×C** | **56 798** | **+1 739** ✅ |
| 10×C | 65 338 | +10 279 |

L'écart résiduel de 1 739 est probablement dû à de légères différences
dans l'ALPE ou les BatchNorm entre notre implémentation et l'article.

---

## 4. Problèmes rencontrés

### Ecart de paramètres avec l'article
**Symptôme** : avec `ffn_dim = 4×C` (convention Transformer standard), on obtient
39 718 params — loin de la cible de 55 059.

**Diagnostic** : en testant systématiquement les multiples de C pour `ffn_dim`,
`8×C` donne le résultat le plus proche (56 798). L'article ne précise pas
cette valeur explicitement — elle a été retrouvée par sanity check.

**Impact** : l'architecture est correcte. L'écart résiduel de 3% n'affecte
pas les shapes ni le comportement du modèle.

---

## 5. Comment utiliser ce module

### Instanciation

```python
from src.mctnet import MCTNet

model_ark = MCTNet(n_classes=5)   # Arkansas
model_cal = MCTNet(n_classes=6)   # Californie
```

### Entraînement

```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_ark.parameters(), lr=0.001)

for X, mask, y in train_loader:
    optimizer.zero_grad()
    logits = model_ark(X, mask)   # (B, 5)
    loss   = criterion(logits, y)
    loss.backward()
    optimizer.step()
```

### Inférence

```python
preds = model_ark.predict(X, mask)   # (B,) — indices des classes
```

### Lancer les tests

```bash
# depuis le dossier "Point 5 — Model Implementation"
python -m tests.test_mctnet
```

---

## 6. Ce qui reste à faire

- 🔲 Implémenter `train.py` — boucle d'entraînement complète + métriques OA/Kappa/F1
- 🔲 Entraîner sur Arkansas et Californie
- 🔲 Comparer OA/Kappa/F1 avec la Table 5 de l'article
