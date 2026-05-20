# Part 2 — Integration of Climate Covariates (ERA5)
## Deep Learning for Crop Classification — M1 SII 2025/2026

---

## Covariables climatiques retenues

Trois covariables ont été sélectionnées depuis le dataset ERA5 Daily (`ECMWF/ERA5/DAILY`) :

| # | Covariable | Source ERA5 | Transformation | Unité finale |
|---|-----------|-------------|----------------|--------------|
| 1 | Température moyenne de l'air | `mean_2m_air_temperature` | − 273.15 | °C |
| 2 | Précipitations totales | `total_precipitation` | × 1000 | mm |
| 3 | Humidité Relative (RH) | `mean_2m_air_temperature` + `dewpoint_2m_temperature` | Formule de Magnus | % (0–100) |

### Formules

**Température (Kelvin → Celsius)**
$$T_{°C} = T_{K} - 273.15$$

**Précipitation (mètres → millimètres)**
$$P_{mm} = P_{m} \times 1000$$

**Humidité Relative — Formule de Magnus**
$$RH = 100 \times \frac{\exp\!\left(\dfrac{17.625 \times T_d}{243.04 + T_d}\right)}{\exp\!\left(\dfrac{17.625 \times T}{243.04 + T}\right)}$$

avec $T$ et $T_d$ en °C, $RH$ clampé entre 0 et 100 %.

### Justifications

- **Température** — La température gouverne directement la phénologie végétale. Chaque culture a un seuil thermique de croissance spécifique (le maïs en Arkansas pic NDVI ~DOY 170, le coton ~DOY 240). C'est le discriminant climatique le plus puissant entre types de cultures.
- **Précipitations** — Le régime pluviométrique est déterminant pour la sélection culturale. Le riz d'Arkansas est cultivé en submersion (~1200 mm/saison), tandis que les amandes et pistaches de Californie poussent en zones semi-arides. Cette variable capture la contrainte hydrique fondamentale.
- **Humidité Relative** — Calculée via la formule de Magnus, elle quantifie le stress hydrique atmosphérique. Elle influence la transpiration foliaire et le développement des cultures, notamment pour les vignes et le coton.

---

## Pipeline complète

```
╔══════════════════════════════════════════════════════════════╗
║  ÉTAPE 1 — GEE : Re-extraire les données avec ERA5          ║
║  (modifier les scripts GEE de l'Étape 2 — Data Acquisition) ║
╚══════════════════════════════════════════════════════════════╝
                          │
                          ▼
╔══════════════════════════════════════════════════════════════╗
║  ÉTAPE 2 — Python : Preprocessing des nouveaux CSV          ║
║  (adapter le preprocessing de l'Étape 4 — Partie 1)        ║
╚══════════════════════════════════════════════════════════════╝
                          │
                          ▼
╔══════════════════════════════════════════════════════════════╗
║  ÉTAPE 3 — Python : Adapter l'architecture MCTNet           ║
║  (modifier l'entrée du modèle pour accepter les covariables) ║
╚══════════════════════════════════════════════════════════════╝
                          │
                          ▼
╔══════════════════════════════════════════════════════════════╗
║  ÉTAPE 4 — Python : Ablation Study                          ║
║  (5 configurations, mêmes hyperparamètres)                  ║
╚══════════════════════════════════════════════════════════════╝
```

---

## ÉTAPE 1 — GEE : Re-extraction avec ERA5

### Pourquoi re-extraire depuis GEE ?

Les CSV produits à l'**Étape 2 — Data Acquisition** (Partie 1) ont été exportés avec `geometries: false`. Il n'y a donc **aucune coordonnée géographique** dans les fichiers, ce qui rend impossible toute jointure spatiale avec ERA5 après coup. La seule solution propre est d'ajouter les bandes ERA5 directement dans GEE avant le `.sample()`.

### Modification à apporter aux scripts Arkansas et California

Ajouter le bloc suivant **après** la définition de `timeSeries` et **avant** la création de `dataset` dans chacun des deux scripts GEE :

#### Arkansas

```javascript
// --- ERA5 : Covariables climatiques ---
var era5 = ee.ImageCollection('ECMWF/ERA5/DAILY')
    .filter(ee.Filter.date('2021-01-01', '2021-12-31'))
    .select([
        'mean_2m_air_temperature',
        'dewpoint_2m_temperature',
        'total_precipitation'
    ]);

var era5Mean = era5.mean().clip(arkAll);

// Variable 1 : Température en Celsius
var temp_celsius = era5Mean.select('mean_2m_air_temperature')
    .subtract(273.15)
    .rename('temp_celsius');

// Variable 2 : Précipitation en mm
var precip_mm = era5Mean.select('total_precipitation')
    .multiply(1000)
    .rename('precip_mm');

// Variable 3 : Humidité Relative (formule de Magnus)
var T  = era5Mean.select('mean_2m_air_temperature').subtract(273.15);
var Td = era5Mean.select('dewpoint_2m_temperature').subtract(273.15);

var rh = ee.Image(100).multiply(
    Td.expression(
        'exp(17.625 * Td / (243.04 + Td)) / exp(17.625 * T / (243.04 + T))',
        {'Td': Td, 'T': T}
    )
).rename('relative_humidity').clamp(0, 100);

var covariates = temp_celsius.addBands(precip_mm).addBands(rh);

// --- Remplacer la ligne dataset originale ---
// AVANT : var dataset = timeSeries.addBands(label).updateMask(combinedMask);
// APRÈS :
var dataset = timeSeries
    .addBands(covariates)
    .addBands(label)
    .updateMask(combinedMask);
```

#### California

```javascript
// Identique, remplacer arkAll par calAll
var era5Cal = ee.ImageCollection('ECMWF/ERA5/DAILY')
    .filter(ee.Filter.date('2021-01-01', '2021-12-31'))
    .select([
        'mean_2m_air_temperature',
        'dewpoint_2m_temperature',
        'total_precipitation'
    ]);

var era5MeanCal = era5Cal.mean().clip(calAll);

var temp_celsius_cal = era5MeanCal.select('mean_2m_air_temperature')
    .subtract(273.15).rename('temp_celsius');

var precip_mm_cal = era5MeanCal.select('total_precipitation')
    .multiply(1000).rename('precip_mm');

var T_cal  = era5MeanCal.select('mean_2m_air_temperature').subtract(273.15);
var Td_cal = era5MeanCal.select('dewpoint_2m_temperature').subtract(273.15);

var rh_cal = ee.Image(100).multiply(
    Td_cal.expression(
        'exp(17.625 * Td / (243.04 + Td)) / exp(17.625 * T / (243.04 + T))',
        {'Td': Td_cal, 'T': T_cal}
    )
).rename('relative_humidity').clamp(0, 100);

var covariates_cal = temp_celsius_cal.addBands(precip_mm_cal).addBands(rh_cal);

var dataset = timeSeries
    .addBands(covariates_cal)
    .addBands(label)
    .updateMask(combinedMask);
```

### Structure des CSV après export

```
363 colonnes au total :
┌──────────────────────┬──────────────┬──────────┬───────────────────┬───────┐
│ B2_t0 ... B12_t35    │ temp_celsius │ precip_mm│ relative_humidity │ label │
│    (360 colonnes)    │              │          │                   │       │
└──────────────────────┴──────────────┴──────────┴───────────────────┴───────┘
```

> **Note sur la résolution** : ERA5 a une résolution native de 27 830 m contre 10 m pour Sentinel-2. GEE gère automatiquement le rééchantillonnage lors du `.sample(scale: 30)` par interpolation bilinéaire. Les pixels d'une même petite zone auront des valeurs ERA5 très proches — c'est normal et attendu.

---

## ÉTAPE 2 — Preprocessing Python

### Ce qui change par rapport à l'Étape 4 — Partie 1

Les 360 colonnes Sentinel-2 sont traitées **exactement comme avant** (même normalisation, même masque, même reshape). Seul ajout : le traitement des 3 nouvelles colonnes ERA5.

### Schéma du preprocessing

```
CSV brut (363 colonnes)
        │
        ├─► Séparer Sentinel (360 col) et ERA5 (3 col)
        │
        ├─► Sur Sentinel (identique Étape 4 — Partie 1) :
        │       reshape (N, 360) → (N, 10, 36)
        │       construire masque AVANT normalisation
        │       mask = (X.sum(axis=1) != 0).astype(float)
        │       normaliser : X / 10 000
        │
        ├─► Sur ERA5 (nouveau) :
        │       temp_celsius      : normalisation min-max → [0, 1]
        │       precip_mm         : normalisation min-max → [0, 1]
        │       relative_humidity : / 100 → [0, 1]
        │
        └─► Sauvegarder en .npy
```

### Fichiers produits

```
data/preprocessed_part2/
├── Arkansas_train_input1.npy   (N, 10, 36)  ← Sentinel (identique Partie 1)
├── Arkansas_train_input2.npy   (N, 36)      ← masque   (identique Partie 1)
├── Arkansas_train_input3.npy   (N, 3)       ← ERA5     (NOUVEAU)
├── Arkansas_train_labels.npy   (N,)
├── Arkansas_val_input1.npy
├── Arkansas_val_input2.npy
├── Arkansas_val_input3.npy
├── Arkansas_val_labels.npy
├── Arkansas_test_input1.npy
├── Arkansas_test_input2.npy
├── Arkansas_test_input3.npy
├── Arkansas_test_labels.npy
└── (même structure pour California)
```

### Code Python

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# CHARGEMENT
# ============================================================

def load_csv(path):
    df = pd.read_csv(path)
    sentinel_cols = [c for c in df.columns if c.startswith('B')]
    era5_cols     = ['temp_celsius', 'precip_mm', 'relative_humidity']
    label_col     = 'label'
    return df[sentinel_cols].values, df[era5_cols].values, df[label_col].values


# ============================================================
# PREPROCESSING SENTINEL (identique Étape 4 — Partie 1)
# ============================================================

def preprocess_sentinel(X_raw):
    # Reshape (N, 360) → (N, 36, 10) → (N, 10, 36)
    N = X_raw.shape[0]
    X = X_raw.reshape(N, 36, 10).transpose(0, 2, 1)

    # Masque construit AVANT normalisation
    mask = (X.sum(axis=1) != 0).astype(np.float32)  # (N, 36)

    # Normalisation
    X = (X / 10000.0).astype(np.float32)

    return X, mask


# ============================================================
# PREPROCESSING ERA5 (nouveau)
# ============================================================

def preprocess_era5(E_raw):
    """
    E_raw : (N, 3) — colonnes : temp_celsius, precip_mm, relative_humidity
    """
    E = E_raw.copy().astype(np.float32)

    # temp_celsius et precip_mm : min-max → [0, 1]
    for i in range(2):
        col_min = E[:, i].min()
        col_max = E[:, i].max()
        if col_max - col_min > 0:
            E[:, i] = (E[:, i] - col_min) / (col_max - col_min)

    # relative_humidity : déjà en % → diviser par 100
    E[:, 2] = E[:, 2] / 100.0
    E[:, 2] = np.clip(E[:, 2], 0.0, 1.0)

    return E


# ============================================================
# SPLIT STRATIFIÉ (identique Étape 4 — Partie 1)
# ============================================================

def stratified_split(X, mask, E, y, n_train_per_class, n_val_per_class, seed=42):
    classes = np.unique(y)
    idx_train, idx_val, idx_test = [], [], []

    for c in classes:
        idx_c = np.where(y == c)[0]
        np.random.seed(seed)
        np.random.shuffle(idx_c)
        idx_train.extend(idx_c[:n_train_per_class])
        idx_val.extend(idx_c[n_train_per_class:n_train_per_class + n_val_per_class])
        idx_test.extend(idx_c[n_train_per_class + n_val_per_class:])

    def get(idx):
        return X[idx], mask[idx], E[idx], y[idx]

    return get(idx_train), get(idx_val), get(idx_test)


# ============================================================
# PIPELINE COMPLÈTE
# ============================================================

def process_region(csv_path, region, n_train, n_val, out_dir):
    print(f"\n=== {region} ===")

    X_raw, E_raw, y = load_csv(csv_path)

    X, mask = preprocess_sentinel(X_raw)
    E       = preprocess_era5(E_raw)

    (X_tr, m_tr, E_tr, y_tr), \
    (X_v,  m_v,  E_v,  y_v),  \
    (X_te, m_te, E_te, y_te) = stratified_split(X, mask, E, y, n_train, n_val)

    for split, (Xi, mi, Ei, yi) in zip(
        ['train', 'val', 'test'],
        [(X_tr, m_tr, E_tr, y_tr),
         (X_v,  m_v,  E_v,  y_v),
         (X_te, m_te, E_te, y_te)]
    ):
        np.save(f'{out_dir}/{region}_{split}_input1.npy', Xi.astype(np.float32))
        np.save(f'{out_dir}/{region}_{split}_input2.npy', mi.astype(np.float32))
        np.save(f'{out_dir}/{region}_{split}_input3.npy', Ei.astype(np.float32))
        np.save(f'{out_dir}/{region}_{split}_labels.npy', yi.astype(np.int64))
        print(f"  {split}: {Xi.shape}, mask: {mi.shape}, ERA5: {Ei.shape}, labels: {yi.shape}")


if __name__ == '__main__':
    import os
    out_dir = '../../data/preprocessed_part2'
    os.makedirs(out_dir, exist_ok=True)

    process_region('Arkansas_10k.csv',   'Arkansas',   n_train=240, n_val=60, out_dir=out_dir)
    process_region('California_10k.csv', 'California', n_train=240, n_val=60, out_dir=out_dir)
```

---

## ÉTAPE 3 — Adapter MCTNet

### Principe : injection tardive des covariables statiques

Les covariables ERA5 sont **statiques** (pas de dimension temporelle). On ne les insère **pas** dans la série temporelle Sentinel. On les injecte **après** le Global Max Pooling, juste avant le MLP classifieur.

```
Input 1 (B, 10, 36) ──► CTFusion ──► CTFusion ──► CTFusion ──► GMP ──► (B, 80)
Input 2 (B, 36)     ──► (masque ALPE)                                        │
                                                                              │ Concat
Input 3 (B, 3)  ──────────────────────────────────────────────────────► (B, 3)
                                                                              │
                                                                         (B, 83)
                                                                              │
                                                                         MLP ──► (B, N_classes)
```

**Pourquoi après le GMP ?**
- Les covariables ERA5 n'ont pas de structure temporelle → les mettre dans les CTFusion stages n'aurait aucun sens
- L'injection après GMP est la modification minimale : elle ne perturbe pas l'architecture existante
- C'est le design standard dans la littérature pour les features statiques dans les modèles temporels

### Modification dans `mctnet.py`

```python
# Dans la classe MCTNet, modifier __init__ et forward :

class MCTNet(nn.Module):
    def __init__(self, n_classes, n_head, kernel_size, dropout,
                 n_covariates=0):          # ← paramètre ajouté
        super().__init__()
        # ... architecture existante inchangée ...

        # MLP : 80 + n_covariates → n_classes
        mlp_input_dim = 80 + n_covariates  # 80 si baseline, 83 si ERA5
        self.classifier = nn.Sequential(
            nn.Linear(mlp_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
        self.n_covariates = n_covariates

    def forward(self, x, mask, covariates=None):
        # ... stages CTFusion existants inchangés ...
        features = self.global_max_pool(x)   # (B, 80)

        if self.n_covariates > 0 and covariates is not None:
            features = torch.cat([features, covariates], dim=1)  # (B, 83)

        return self.classifier(features)
```

### Modification dans `train.py`

```python
# Modifier CropDataset pour charger input3 si disponible

class CropDataset(Dataset):
    def __init__(self, data_dir, region, split, use_covariates=False):
        prefix = os.path.join(data_dir, f'{region}_{split}')
        self.X    = torch.from_numpy(np.load(f'{prefix}_input1.npy')).float()
        self.mask = torch.from_numpy(np.load(f'{prefix}_input2.npy')).float()
        self.y    = torch.from_numpy(np.load(f'{prefix}_labels.npy')).long()
        self.use_covariates = use_covariates

        if use_covariates:
            self.cov = torch.from_numpy(np.load(f'{prefix}_input3.npy')).float()

    def __getitem__(self, idx):
        if self.use_covariates:
            return self.X[idx], self.mask[idx], self.cov[idx], self.y[idx]
        return self.X[idx], self.mask[idx], self.y[idx]
```

---

## ÉTAPE 4 — Ablation Study

### 5 configurations à entraîner

| Config | Description | Input 3 (covariates) | `n_covariates` |
|--------|-------------|----------------------|----------------|
| 1 | Sentinel seul *(baseline)* | — | 0 |
| 2 | Sentinel + température | `temp_celsius` uniquement | 1 |
| 3 | Sentinel + précipitations | `precip_mm` uniquement | 1 |
| 4 | Sentinel + humidité relative | `relative_humidity` uniquement | 1 |
| 5 | Sentinel + toutes covariables | `[temp, precip, RH]` | 3 |

> **Règle stricte** : mêmes hyperparamètres, même split, même `random_state=42` pour toutes les configs.

### Métriques à reporter

Pour chaque config et chaque région (Arkansas, California) :

| Métrique | Description |
|----------|-------------|
| OA | Overall Accuracy |
| Kappa | Cohen's Kappa |
| F1 | F1 macro-averaged |

### Tableau de résultats attendu

```
Config                  │ Arkansas OA │ Kappa │ F1   │ California OA │ Kappa │ F1
────────────────────────┼─────────────┼───────┼──────┼───────────────┼───────┼──────
1. Sentinel seul        │             │       │      │               │       │
2. + Température        │             │       │      │               │       │
3. + Précipitations     │             │       │      │               │       │
4. + Humidité Relative  │             │       │      │               │       │
5. + Tout ERA5          │             │       │      │               │       │
```

---


