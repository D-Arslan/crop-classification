> **Internal working notes, April 2026** (in French). Written during development and kept for history;
> code, paths and results may differ from the final version. See the root [README](../../../README.md).

# Documentation — Entraînement MCTNet
## `train.py`

---

## Table des matières

1. [Contexte et rôle](#1-contexte-et-rôle)
2. [Structure du fichier](#2-structure-du-fichier)
3. [CropDataset](#3-cropdataset)
4. [Boucle d'entraînement](#4-boucle-dentraînement)
5. [Métriques d'évaluation](#5-métriques-dévaluation)
6. [Décisions d'implémentation](#6-décisions-dimplémentation)
7. [Comment lancer l'entraînement](#7-comment-lancer-lentraînement)

---

## 1. Contexte et rôle

`train.py` est le script d'entraînement principal. Il assemble tous les modules
(`CropDataset`, `MCTNet`, `evaluate`) pour produire un modèle entraîné et les
métriques finales comparables à la Table 5 de l'article.

---

## 2. Structure du fichier

```
train.py
├── CONFIG            ← tous les hyperparamètres en un seul endroit
├── N_CLASSES         ← {Arkansas: 5, California: 6}
├── CropDataset       ← charge les .npy de Sarah
├── train_one_epoch   ← une époque d'entraînement
├── evaluate          ← OA / Kappa / F1 sur un split
└── main              ← boucle complète + sauvegarde + résultats finaux
```

---

## 3. `CropDataset`

Charge les fichiers `.npy` produits par le preprocessing (Point 4 — Sarah).

```python
dataset = CropDataset(data_dir='../../data/preprocessed/scale30', region='Arkansas', split='train')
# Retourne à chaque __getitem__ : (X, mask, y)
# X    : (10, 36)  float32
# mask : (36,)     float32
# y    : ()        long
```

Les types sont garantis par le Dataset même si les fichiers sont régénérés :
```python
self.X    = self.X.float()
self.mask = self.mask.float()
self.y    = self.y.long()
```

---

## 4. Boucle d'entraînement

### Hyperparamètres (Table 3 de l'article)

```python
CONFIG = {
    'lr':          0.001,
    'batch_size':  32,
    'epochs':      200,
    'n_head':      5,
    'kernel_size': 3,
    'dropout':     0.1,
    'print_every': 10,
}
```

### Progression typique

```
Epoch   1/200 | train_loss=1.6094 | val_OA=0.20 | val_F1=0.20 | best_F1=0.20
Epoch  10/200 | train_loss=0.8500 | val_OA=0.65 | val_F1=0.62 | best_F1=0.64
Epoch  50/200 | train_loss=0.4200 | val_OA=0.85 | val_F1=0.83 | best_F1=0.85
...
Epoch 200/200 | train_loss=0.1800 | val_OA=0.96 | val_F1=0.93 | best_F1=0.93
```

---

## 5. Métriques d'évaluation

Trois métriques calculées avec `sklearn`, conformes à l'article :

| Métrique | Fonction sklearn | Description |
|----------|-----------------|-------------|
| OA (Overall Accuracy) | `accuracy_score` | % de pixels correctement classés |
| Kappa | `cohen_kappa_score` | Accord au-delà du hasard (0=hasard, 1=parfait) |
| F1 macro | `f1_score(average='macro')` | Moyenne des F1 par classe — sensible au déséquilibre |

Le F1 macro est la métrique principale utilisée pour sauvegarder le meilleur modèle.

### Résultats cibles (Table 5 de l'article)

| Zone | OA | Kappa | F1 |
|------|----|-------|----|
| Arkansas | 0.968 | 0.951 | 0.933 |
| Californie | 0.852 | 0.806 | 0.829 |

---

## 6. Décisions d'implémentation

### F1 macro comme critère de sauvegarde
Le meilleur modèle est sauvegardé selon le **F1 macro** (pas l'OA). Raison : avec des classes déséquilibrées (Soybeans = 45% en Arkansas), l'OA peut être élevée même si les petites classes sont mal classées. Le F1 macro pénalise les mauvaises performances sur les petites classes.

### Learning rate scheduler
L'article ne mentionne pas de scheduler. Le code utilise Adam (`lr=0.001`, Table 3) avec `ReduceLROnPlateau` (facteur 0.5, patience 10 sur la loss val) et un early stopping (patience 20) — écart à l'article.

### CrossEntropyLoss sans class weights
L'article n'utilise pas de class weights. La version pondérée est documentée dans `idees_partie3.md` comme piste d'amélioration pour les classes déséquilibrées.

### Softmax absent en training
`nn.CrossEntropyLoss` intègre LogSoftmax + NLLLoss — pas besoin de Softmax explicite dans `forward()`. En inférence, `predict()` appelle directement `argmax` sur les logits.

### Sauvegarde du meilleur modèle
Seul le meilleur modèle (selon F1 val) est sauvegardé — `best_{region}_{model}_seed{seed}.pth`. L'évaluation finale sur le test set utilise ce meilleur modèle, pas le dernier.

---

## 7. Comment lancer l'entraînement

```bash
# Depuis le dossier "Point 5 — Model Implementation"

# Arkansas
python train.py --region Arkansas --data_dir ../../data/preprocessed/scale30

# Californie
python train.py --region California --data_dir ../../data/preprocessed/scale30
```

### Sortie attendue (fin d'entraînement)

```
============================================================
RESULTATS FINAUX — Arkansas
============================================================
  OA    : 0.xxxx  (article, Table 5 : 0.968)
  Kappa : 0.xxxx  (article, Table 5 : 0.951)
  F1    : 0.xxxx  (article, Table 5 : 0.933)

Modele sauvegarde : best_Arkansas_mctnet_seed42.pth
Metriques ecrites : results/metrics_Arkansas_mctnet_seed42.json
```

### Prérequis

Les fichiers `.npy` doivent être présents dans `data/preprocessed/scale30/` à la racine du projet.
Voir `README.md` pour les instructions de récupération depuis le Drive partagé.
