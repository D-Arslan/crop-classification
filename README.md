# Crop Classification — MCTNet
## M1 SII — USTHB — 2025/2026

Reproduction de l'article :
> Wang et al., 2024 — *"A lightweight CNN-Transformer network for pixel-based crop mapping using time-series Sentinel-2 imagery"*, Computers and Electronics in Agriculture

---

## Équipe

| Membre | Rôle |
|--------|------|
| Arslan | ALPE + Transformer sub-module |
| Tesnime | CNN sub-module |
| Sarah | Data Preprocessing |

---

## Structure du projet

```
crop-classification/
├── avancement.md                          ← état d'avancement + workflow Git
├── Partie I/
│   ├── resume.md                          ← documentation technique complète
│   ├── Point 1 — Literature Review/
│   ├── Point 2 — Dataset Acquisition/     ← scripts GEE + merge Python
│   ├── Point 3 — Data Exploration/        ← notebook d'exploration
│   ├── Point 4 — Data Preprocessing/      ← notebook preprocessing
│   └── Point 5 — Model Implementation/
│       ├── src/                           ← modules PyTorch
│       │   └── transformer_alpe.py
│       └── tests/                         ← tests par module
│           └── test_transformer.py
```

---

## Prérequis

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

---

## Données

Les fichiers de données ne sont pas versionnés dans ce repo (taille).
Ils sont disponibles sur le Google Drive partagé de l'équipe.

### Récupérer les données

1. Télécharger le dossier `data/` depuis le Google Drive partagé
2. Le placer à la racine du projet :

```
crop-classification/
└── data/
    ├── raw/
    │   ├── Arkansas_10k.csv
    │   └── California_10k.csv
    └── preprocessed/
        ├── Arkansas_train_input1.npy
        ├── Arkansas_train_input2.npy
        ├── Arkansas_train_labels.npy
        ├── Arkansas_val_input1.npy
        ├── Arkansas_val_input2.npy
        ├── Arkansas_val_labels.npy
        ├── Arkansas_test_input1.npy
        ├── Arkansas_test_input2.npy
        ├── Arkansas_test_labels.npy
        ├── California_train_input1.npy
        ├── California_train_input2.npy
        ├── California_train_labels.npy
        ├── California_val_input1.npy
        ├── California_val_input2.npy
        ├── California_val_labels.npy
        ├── California_test_input1.npy
        ├── California_test_input2.npy
        └── California_test_labels.npy
```

### Régénérer les fichiers preprocessés (optionnel)

Si tu veux relancer le preprocessing depuis les CSV bruts :

```bash
# Ouvrir et exécuter toutes les cellules du notebook
jupyter notebook "Partie I/Point 4 — Data Preprocessing/preprocessing_MCTNet_N_10_36_20.ipynb"
```

Les fichiers `.npy` seront générés dans `Partie I/Point 4 — Data Preprocessing/preprocessed/`.

---

## Lancer les tests

```bash
# Depuis le dossier Point 5
cd "Partie I/Point 5 — Model Implementation"

# Transformer + ALPE
python -m tests.test_transformer

# CNN (quand disponible)
python -m tests.test_cnn
```

---

## Workflow Git

Voir [avancement.md](avancement.md) pour le détail complet.

```
main          ← stable, remise prof
└── dev       ← intégration validée
    ├── arslan/transformer
    ├── tesnime/cnn
    └── sarah/preprocessing
```

**Règle** : on ne push jamais directement sur `main` ou `dev`.
Chaque modification passe par une Pull Request.
