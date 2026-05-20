# Point 4 — Data Preprocessing

## Statut : ✅ Terminé

## Objectif
Transformer les CSV bruts (Arkansas, Californie) en tenseurs NumPy prêts à être consommés par MCTNet. Deux versions ont été produites correspondant aux deux résolutions testées : **scale 30m** et **scale 20m**.

## Fichiers produits

### Notebooks
- `preprocessing_MCTNet_N_10_36_20.ipynb` — version scale 20m
- `preprocessing_30.ipynb` — version scale 30m (alias `preprocessing_MCTNet_corrected_30_2.ipynb`)
- `data_correction.ipynb` — correction et fusion des CSV bruts Arkansas (zones Z1–Z5)

### Données .npy — deux versions

Les fichiers sont dans `data/preprocessed/scale30/` et `data/preprocessed/scale20/` à la racine du projet.

**Par région et split (× 2 scales = 24 fichiers chacun) :**
| Fichier | Shape | Description |
|---------|-------|-------------|
| `{Region}_{split}_input1.npy` | (N, 10, 36) | Features spectrales normalisées |
| `{Region}_{split}_input2.npy` | (N, 36) | Masque de données manquantes {0,1} |
| `{Region}_{split}_labels.npy` | (N,) | Labels de classe entiers |
| `{Region}_input1.npy` | (10000, 10, 36) | Dataset complet (non splitté) |
| `{Region}_input2.npy` | (10000, 36) | Masque complet |
| `{Region}_labels.npy` | (10000,) | Labels complets |

**Splits Arkansas (5 classes, 10 000 pixels équilibrés — 2000/classe) :**
| Split | Shape input1 | Shape input2 | Shape labels |
|-------|-------------|-------------|-------------|
| train | (1200, 10, 36) | (1200, 36) | (1200,) |
| val   | (300, 10, 36)  | (300, 36)  | (300,)  |
| test  | (8500, 10, 36) | (8500, 36) | (8500,) |

**Splits Californie (6 classes, 10 000 pixels) :**
| Split | Shape input1  | Shape input2 | Shape labels |
|-------|--------------|-------------|-------------|
| train | (1440, 10, 36) | (1440, 36) | (1440,) |
| val   | (360, 10, 36)  | (360, 36)  | (360,)  |
| test  | (8200, 10, 36) | (8200, 36) | (8200,) |

---

## Specs d'interface — contrat avec Point 5

### Tenseur X — Input 1
| Propriété | Valeur |
|-----------|--------|
| Forme | `(B, 10, 36)` — batch × bandes × timesteps |
| Type | `float32` |
| Ordre des bandes | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 (index 0 à 9) |
| Normalisation | Division par 10 000 (`X / 10000`) |
| Plage | [0.0, ~1.30] — valeurs légèrement > 1 possibles (réflectances anormales/nuages, normal) |

### Tenseur M — Input 2 (masque)
| Propriété | Valeur |
|-----------|--------|
| Forme | `(B, 36)` — batch × timesteps |
| Type | `float32` |
| Valeurs | `1.0` = timestep valide, `0.0` = timestep manquant |
| Construction | `mask = ~(X_raw == 0).all(axis=1)` — un timestep est manquant si **toutes** ses 10 bandes valent 0 |

> **Important** : le masque est construit **avant** la normalisation, sur les valeurs brutes.

### Labels Y
| Propriété | Valeur |
|-----------|--------|
| Forme | `(B,)` |
| Type | `int64` (compatible `nn.CrossEntropyLoss`) |
| Valeurs | entiers 0 à N_classes-1 (inchangés par rapport aux CSV) |

### Interface DataLoader attendue
```python
for X, mask, y in dataloader:
    # X    : (B, 10, 36)  float32
    # mask : (B, 36)      float32
    # y    : (B,)         long
```

---

## Détails techniques

### Normalisation
Division par 10 000 (`X / 10000.0`). Les réflectances Sentinel-2 L2A sont stockées en entiers × 10 000.
Aucun Z-Score ni Min-Max — non mentionné dans l'article (Section 2.2.4).
Quelques valeurs légèrement > 1.0 sont normales (pixels bruités/nuageux résiduels).

### Gestion des données manquantes
Les 0 sont conservés sans imputation. GEE marque un timestep manquant en mettant toutes ses bandes à 0 via `unmask(0)`. Le masque est construit sur les valeurs brutes, avant normalisation. Après normalisation, ces positions restent à 0.0. Aucune interpolation appliquée — le module ALPE gère les manquants.

| Région | Taux de manquants | Pixels avec 0 manquant |
|--------|------------------|------------------------|
| Arkansas | 22.71% | 0 (0.0%) |
| Californie | 16.13% | 4 (0.04%) |

### Reshape et ordre des axes
```
CSV brut    : (N, 360) — colonnes spectrales à plat
→ reshape   : (N, 36, 10) — (pixels, timesteps, bandes)
→ transpose : (N, 10, 36) — (pixels, bandes, timesteps)  ← format attendu par MCTNet
```
Format channels-first requis par les Conv1D du modèle (Figure 3 de l'article : matrice 10×36, bandes × temps).

### Split
Stratifié par classe, `n_per_class=300`, `val_ratio=0.2`, `random_state=42`.
- 300 par classe : 240 train + 60 val, reste → test
- Reproductible avec `numpy.random.seed(42)` + `train_test_split(random_state=42)`

### Correction Arkansas
Les CSV bruts Arkansas (zones Z1–Z5) avaient des fichiers vides (Z2, Z5 de certaines répétitions). Corrigés dans `data_correction.ipynb` avec gestion `EmptyDataError`. Dataset final : **10 000 pixels équilibrés à 2000/classe** (contrairement à la version précédente à 12 500 pixels déséquilibrés).

---

## Problèmes rencontrés & solutions

| Problème | Solution |
|----------|----------|
| Fichiers CSV Arkansas zones vides (EmptyDataError) | `try/except EmptyDataError` dans `data_correction.ipynb` |
| Dataset Arkansas déséquilibré (ancienne version) | Re-échantillonnage à 2000 pixels/classe → 10 000 total équilibré |
| Valeurs > 1.0 après normalisation | Normal pour Sentinel-2 — quelques pixels > 10 000 en brut (nuages résiduels) |

---

## Décisions prises & pourquoi

| Décision | Raison |
|----------|--------|
| Division par 10 000 | Convention Sentinel-2 L2A, non-précisé dans l'article |
| Pas d'imputation des manquants | Le module ALPE est conçu pour les gérer |
| Masque construit avant normalisation | La normalisation ne modifie pas les 0, mais par précaution |
| 2000 pixels/classe pour Arkansas | Équilibre les classes (ancienne version avait 45% Soybeans) |
| Deux scales (30m et 20m) | Comparaison expérimentale — permet d'évaluer l'impact de la résolution |

---

## Lien avec les autres points
- ← **Point 2** : consomme `arkansas_final.csv` et `California_10k.csv`
- ← **Point 3** : l'exploration a confirmé que les 0 sont des manquants (pas des vraies valeurs spectrales)
- → **Point 5** : fournit les tenseurs `(X, mask, y)` dans `data/preprocessed/scale30/` et `data/preprocessed/scale20/`
