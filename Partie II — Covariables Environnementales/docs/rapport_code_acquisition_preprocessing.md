# Rapport de projet — Code d'acquisition et de prétraitement
## Cartographie des cultures avec MCTNet — Pipeline complet

---

> Ce rapport explique en détail le fonctionnement des deux scripts du projet :
> - **Script GEE** (`Nouveau_Document_texte.txt`) : acquisition des données Sentinel-2 via Google Earth Engine
> - **Notebook Python** (`preprocessing_MCTNet_N_10_36_20.ipynb`) : prétraitement des données pour MCTNet

---

## PARTIE 1 — Acquisition des données sur Google Earth Engine (GEE)

Google Earth Engine est une plateforme cloud qui permet d'accéder à des archives satellitaires massives et d'effectuer des traitements directement sur les serveurs de Google, sans télécharger les images brutes.

---

### 1.1 Définition des zones géographiques

```javascript
var ark1 = ee.Geometry.Rectangle([-91.75, 34.40, -91.25, 34.80]);
var ark2 = ee.Geometry.Rectangle([-92.10, 33.80, -91.60, 34.20]);
...
var arkAll = ark1.union(ark2).union(ark3).union(ark4).union(ark5);
```

**Rôle :** Cinq rectangles géographiques sont définis pour couvrir l'Arkansas. Chaque rectangle est défini par ses coordonnées `[longitude_min, latitude_min, longitude_max, latitude_max]`. La variable `arkAll` représente l'union des 5 zones, utilisée comme emprise globale pour filtrer les images.

---

### 1.2 Sélection des bandes spectrales

```javascript
var bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];
```

**Rôle :** On sélectionne les 10 bandes Sentinel-2 retenues dans l'article (les bandes 1, 9 et 10 sont exclues car leur résolution de 60 m est insuffisante). Ces 10 bandes couvrent le visible (B2, B3, B4), les bords rouges (B5–B8A), le proche infrarouge (B8) et l'infrarouge à ondes courtes (B11, B12).

---

### 1.3 Fonction de masquage des nuages

```javascript
function maskClouds(image) {
  var qa = image.select('QA60');
  var mask = qa.bitwiseAnd(1 << 10).eq(0)
                .and(qa.bitwiseAnd(1 << 11).eq(0));
  return image.updateMask(mask);
}
```

**Rôle :** Cette fonction utilise la bande `QA60` (Quality Assessment) fournie avec chaque image Sentinel-2. Elle contient des indicateurs binaires encodés bit par bit :
- **Bit 10** = présence de nuages opaques
- **Bit 11** = présence de cirrus (nuages fins)

L'opération `bitwiseAnd` isole ces bits. Si l'un des deux est activé, le pixel est masqué (rendu transparent). `updateMask` applique ce masque à l'image, rendant les pixels nuageux invisibles pour les calculs suivants.

---

### 1.4 Image de sécurité (astuce anti-collection vide)

```javascript
var safetyImage = ee.Image.constant([0,0,0,0,0,0,0,0,0,0])
  .rename(bands).toFloat()
  .selfMask();
```

**Rôle :** C'est une astuce technique importante. Si aucune image Sentinel-2 n'est disponible pour une fenêtre de 10 jours (nuages permanents sur toute la période), la collection serait vide et le calcul de la médiane échouerait. On ajoute donc systématiquement une image de zéros **masquée** (`selfMask()` rend les zéros transparents) à chaque collection. La médiane ignorera cette image si d'autres observations valides existent. Si c'est la seule image, la médiane retournera 0 (valeur manquante).

---

### 1.5 Fonction de construction d'un composite 10 jours

```javascript
function getComposite(startDay) {
  var start = ee.Date('2021-01-01').advance(startDay, 'day');
  var end = start.advance(10, 'day');
  
  var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(arkAll)
    .filterDate(start, end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(maskClouds)
    .select(bands)
    .map(function(img) { return img.toFloat(); });
  
  var safeColl = collection.merge(ee.ImageCollection([safetyImage]));
  return safeColl.median().unmask(0).toFloat();
}
```

**Rôle étape par étape :**

| Étape | Code | Explication |
|-------|------|-------------|
| Définir la fenêtre | `advance(startDay, 'day')` | Calcule la date de début et de fin de la fenêtre de 10 jours |
| Filtrer par zone | `filterBounds(arkAll)` | Ne garde que les images qui couvrent les zones d'étude |
| Filtrer par date | `filterDate(start, end)` | Ne garde que les images dans la fenêtre de 10 jours |
| Pré-filtrer les nuages | `filter(lt('CLOUDY_PIXEL_PERCENTAGE', 20))` | Élimine les images avec plus de 20 % de couverture nuageuse globale |
| Masquer pixel par pixel | `map(maskClouds)` | Applique le masque nuage fin sur chaque pixel de chaque image restante |
| Calculer la médiane | `median()` | Calcule la valeur médiane de toutes les observations valides pixel par pixel |
| Remplacer les vides | `unmask(0)` | Les pixels sans aucune observation valide reçoivent la valeur 0 (= donnée manquante) |

---

### 1.6 Génération des 36 composites et empilement

```javascript
var t0  = getComposite(0);   var t1  = getComposite(10);
...
var t35 = getComposite(350);

var timeSeries = t0.addBands(t1).addBands(t2)...addBands(t35);
```

**Rôle :** On appelle `getComposite` 36 fois, une par fenêtre de 10 jours (0, 10, 20, ..., 350 jours depuis le 1er janvier 2021). Chaque composite est une image à 10 bandes. On les empile toutes avec `addBands`, créant une image finale à **360 bandes** (10 bandes × 36 temps). Chaque pixel de cette image contient donc toute la série temporelle d'une année.

---

### 1.7 Vérité terrain et masques

```javascript
var cdl_full   = ee.Image('USDA/NASS/CDL/2021');
var cropland   = cdl_full.select('cropland');
var confidence = cdl_full.select('confidence');
var confMask   = confidence.gte(95);

var label = ee.Image(4).toInt().rename('label');
label = label.where(cropland.eq(1), 0);   // Corn
label = label.where(cropland.eq(2), 1);   // Cotton
label = label.where(cropland.eq(3), 2);   // Rice
label = label.where(cropland.eq(5), 3);   // Soybeans
```

**Rôle :** Le CDL (Cropland Data Layer) fournit l'étiquette de culture pour chaque pixel. On filtre à une confiance ≥ 95 % pour ne garder que les pixels dont la classification est fiable. Les codes CDL (1=maïs, 2=coton, etc.) sont remapiés en indices 0–4. Tous les autres types de cultures reçoivent l'étiquette 4 ("Others").

```javascript
var worldCover   = ee.Image('ESA/WorldCover/v200/2021').select('Map');
var croplandMask = worldCover.eq(40);
var combinedMask = agMask.and(confMask).and(croplandMask);
```

**Rôle :** Le masque combiné s'assure que les échantillons proviennent uniquement de pixels qui sont :
- classifiés comme cultures agricoles dans le CDL (`agMask`)
- avec une confiance ≥ 95 % (`confMask`)
- classifiés comme terres cultivées dans ESA WorldCover (`croplandMask`)

---

### 1.8 Échantillonnage stratifié et export

```javascript
var samples = dataset.clip(zones[z]).stratifiedSample({
    numPoints   : PER_CLASS,
    classBand   : 'label',
    classValues : [0, 1, 2, 3, 4],
    classPoints : [PER_CLASS, PER_CLASS, PER_CLASS, PER_CLASS, PER_CLASS],
    scale       : 20,
    seed        : 42
});

Export.table.toDrive({
    collection : samples,
    fileFormat : 'CSV'
});
```

**Rôle :** `stratifiedSample` tire aléatoirement `PER_CLASS = 400` points par classe et par zone, garantissant un échantillonnage équilibré entre les classes. Le paramètre `seed=42` assure la reproductibilité. Les résultats sont exportés en fichiers CSV vers Google Drive, un fichier par zone (5 zones × 5 classes = 25 exports qui seront fusionnés ensuite).

---

## PARTIE 2 — Prétraitement Python (Notebook)

---

### 2.1 Imports et configuration

```python
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

ARK_CSV = "arkansas_final.csv"
CAL_CSV = "California_10k_20m.csv"
BANDS   = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
N_TIMES = 36
N_BANDS = 10
SCALE   = 10000.0
```

**Rôle :** On définit toutes les constantes du projet en un seul endroit pour faciliter les modifications futures. `SCALE = 10000.0` vient du fait que Sentinel-2 stocke ses réflectances en entiers multipliés par 10 000 (ex : une réflectance de 0.06 est stockée comme 600). On devra donc diviser par 10 000 pour obtenir les valeurs physiques entre 0 et 1.

---

### 2.2 Construction des noms de colonnes

```python
def build_spectral_cols():
    cols = []
    for t in range(N_TIMES):
        for band in BANDS:
            cols.append(band if t == 0 else f'{band}_{t}')
    return cols
```

**Rôle :** GEE nomme les bandes selon une convention particulière. Le premier pas de temps (t=0) garde le nom original (`B2`, `B3`...), et les suivants reçoivent un suffixe (`B2_1`, `B2_2`..., `B2_35`). Cette fonction reconstruit exactement ces 360 noms de colonnes pour lire correctement le CSV exporté depuis GEE.

```
Exemple de sortie :
  t=0  : ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
  t=1  : ['B2_1', 'B3_1', ..., 'B12_1']
  t=35 : ['B2_35', 'B3_35', ..., 'B12_35']
```

---

### 2.3 Chargement et inspection des CSV

```python
ark_df = pd.read_csv(ARK_CSV)
ark_df = ark_df.drop(columns=['system:index', '.geo'], errors='ignore')
```

**Rôle :** On charge les CSV et on supprime les colonnes techniques ajoutées par GEE (`system:index` = identifiant interne, `.geo` = coordonnées géographiques) qui ne sont pas utiles pour l'entraînement du modèle. On vérifie ensuite la distribution des classes et l'absence de valeurs NaN dans les colonnes spectrales.

---

### 2.4 Extraction brute → shape (N, 10, 36)

```python
def extract_raw(df):
    X = df[spectral_cols].values.astype(np.float32)   # (N, 360)
    X = X.reshape(len(X), N_TIMES, N_BANDS)            # (N, 36, 10)
    X = X.transpose(0, 2, 1)                           # (N, 10, 36)
    return X
```

**Rôle :** C'est une étape cruciale de restructuration des données. Le CSV contient les 360 valeurs par ligne dans l'ordre `t0_B2, t0_B3... t0_B12, t1_B2...`. On les réorganise en une matrice 3D :

```
Étape 1 — Lecture plate :
  (N, 360) : N pixels, 360 valeurs dans l'ordre t0_B2...t35_B12

Étape 2 — Reshape :
  (N, 36, 10) : N pixels, 36 pas de temps, 10 bandes

Étape 3 — Transpose :
  (N, 10, 36) : N pixels, 10 bandes, 36 pas de temps
                ↑ Format "channels-first" attendu par MCTNet
```

Le format `(N, 10, 36)` est appelé **channels-first** car les bandes (canaux) sont en deuxième dimension, avant le temps. C'est le format standard pour les convolutions 1D en PyTorch.

---

### 2.5 Construction du masque Input 2 → shape (N, 36)

```python
def build_mask(X_raw):
    all_zero = (X_raw == 0).all(axis=1)        # (N, 36) : True si toutes les bandes = 0
    mask = (~all_zero).astype(np.float32)       # 1 = observation valide, 0 = manquante
    return mask
```

**Rôle :** C'est la construction de l'**Input 2** décrit dans l'article, le masque des données manquantes. Pour chaque pixel et chaque pas de temps, on vérifie si **toutes** les 10 bandes valent 0 simultanément. Si oui, ce pas de temps est considéré comme manquant (valeur assignée par GEE quand aucune observation n'était disponible).

```
Exemple pour un pixel :
  t=0  : [0.06, 0.08, 0.05, ...] → au moins une valeur non nulle → mask = 1 (valide)
  t=5  : [0.00, 0.00, 0.00, ...] → tout à zéro → mask = 0 (manquant)
  t=22 : [0.12, 0.15, 0.11, ...] → valide → mask = 1

Résultat : [1, 1, 1, 1, 1, 0, ..., 1, 1, 0, ..., 1]  (vecteur de 36 valeurs)
```

Ce masque sera utilisé par le module ALPE pour indiquer au modèle quels pas de temps sont fiables.

---

### 2.6 Normalisation

```python
def normalize(X_raw):
    return (X_raw / SCALE).astype(np.float32)   # Division par 10 000
```

**Rôle :** Sentinel-2 stocke ses réflectances sous forme d'entiers en multipliant par 10 000 pour éviter les décimales. On divise ici par `SCALE = 10000.0` pour retrouver les réflectances physiques dans l'intervalle `[0, 1]`. Les valeurs 0 (données manquantes) restent à 0 après normalisation, ce qui est voulu. Cette normalisation correspond exactement à ce qui est décrit dans la section 2.2.4 du paper.

---

### 2.7 Extraction des labels

```python
ark_labels = ark_df['label'].values.astype(np.int64)
```

**Rôle :** On extrait simplement la colonne `label` (0 à 4 pour l'Arkansas, 0 à 5 pour la Californie) comme tableau NumPy d'entiers. Ces entiers seront utilisés comme cibles lors de l'entraînement du modèle (classification multi-classes).

---

### 2.8 Vérification d'intégrité

```python
def check_integrity(input1, input2, labels, label_map, name):
    # Vérifications de forme
    if input1.shape != (N, N_BANDS, N_TIMES): errors.append(...)
    if input2.shape != (N, N_TIMES): errors.append(...)
    # Vérifications numériques
    if np.isnan(input1).any(): errors.append(...)
    if np.isinf(input1).any(): errors.append(...)
    # Cohérence masque/données
    masked_vals = input1[missing_3d]
    n_nonzero = (masked_vals != 0).sum()
```

**Rôle :** Cette cellule agit comme un **test de qualité automatique** avant de continuer. Elle vérifie :
- Que les shapes sont exactement `(N, 10, 36)` et `(N, 36)` comme attendu par MCTNet
- Qu'il n'existe aucune valeur NaN (Not a Number) ou infinie qui ferait planter l'entraînement
- Que le masque ne contient que des 0 et des 1
- La **cohérence interne** : les positions marquées comme manquantes dans le masque (= 0) doivent correspondre à des valeurs nulles dans Input1. Toute incohérence est signalée comme avertissement.

---

### 2.9 Split Train / Validation / Test

```python
def split_dataset(input1, input2, labels, ...):
    for cls in sorted(label_map.keys()):
        idx_cls = np.where(labels == cls)[0]
        n_use   = min(n_per_class, len(idx_cls))   # max 300 par classe
        np.random.shuffle(idx_cls)
        idx_trainval = idx_cls[:n_use]              # 300 pour train+val
        idx_test     = idx_cls[n_use:]              # reste pour test
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=0.2, random_state=42
        )
```

**Rôle :** Cette fonction reproduit exactement le protocole du **Tableau 2 de l'article**. Pour chaque classe séparément (stratification), on sélectionne 300 échantillons qui sont divisés en train (240) et validation (60) selon un ratio 80/20. Tous les échantillons restants constituent l'ensemble de test. Cette approche garantit que :
- Le test représente la vraie distribution géographique (beaucoup plus d'échantillons)
- Chaque classe est également représentée en train/val
- Les résultats sont reproductibles grâce à `random_state=42`

```
Arkansas résultant :
  Train  : 240 × 5 classes = 1 200 échantillons
  Val    : 60  × 5 classes =   300 échantillons
  Test   : reste           ≈ 8 500 échantillons
```

---

### 2.10 Sauvegarde

```python
np.save(f'{region}_input1.npy', input1)    # (N, 10, 36) — données spectrales normalisées
np.save(f'{region}_input2.npy', input2)    # (N, 36) — masque données manquantes
np.save(f'{region}_labels.npy', labels)    # (N,) — étiquettes des cultures
# + les 3 splits séparément
np.save(f'{region}_train_input1.npy', train['input1'])
...
```

**Rôle :** On sauvegarde les données au format `.npy` (format binaire NumPy), qui est très rapide à charger et compact. On sauvegarde à la fois le dataset complet et les trois splits séparément pour faciliter le chargement lors de l'entraînement du modèle. Au total, **12 fichiers** sont générés (2 régions × 3 types × 2 splits + complets).

---

### 2.11 Visualisations

**Taux de données manquantes par pas de temps :**
```python
miss_pct = (M_data == 0).mean(axis=0) * 100    # (36,) — un taux par fenêtre temporelle
ax.bar(days, miss_pct, ...)
```
Permet de visualiser quelles périodes de l'année souffrent le plus de la couverture nuageuse (en général l'hiver pour l'Arkansas).

**Profils NDVI :**
```python
NIR  = X_raw[:, IDX['B8'], :]
Red  = X_raw[:, IDX['B4'], :]
ndvi = (NIR - Red) / (NIR + Red + eps)
```
Reproduit la Figure 2 de l'article. Le NDVI est calculé uniquement pour la **visualisation**, il n'est pas utilisé comme feature d'entrée du modèle (qui reçoit les 10 bandes brutes).

**Distribution des bandes :**
```python
band_data    = ark_input1[:, i, :]       # (N, 36) pour la bande i
valid_pixels = band_data[ark_mask == 1]  # on exclut les données manquantes
ax.hist(valid_pixels, bins=60, ...)
```
Permet de vérifier que la normalisation est correcte (toutes les valeurs entre 0 et 1) et de détecter d'éventuelles anomalies dans les données.

---

## 3. Résumé du pipeline complet

```
GOOGLE EARTH ENGINE
        │
        ├─ 1. Définition des 5 zones Arkansas
        ├─ 2. Filtrage Sentinel-2 (nuages < 20%)
        ├─ 3. Masquage pixel par pixel (QA60)
        ├─ 4. Médiane par fenêtre de 10 jours (36 fenêtres)
        ├─ 5. Empilement → image 360 bandes
        ├─ 6. Étiquetage CDL + filtres qualité
        └─ 7. Export CSV (400 pts/classe/zone)
                        │
                        ▼
PYTHON (NOTEBOOK)
        │
        ├─ 8.  Lecture CSV + suppression colonnes GEE
        ├─ 9.  Reconstruction noms colonnes (convention GEE)
        ├─ 10. Reshape : (N, 360) → (N, 10, 36) [channels-first]
        ├─ 11. Masque données manquantes → Input 2 (N, 36)
        ├─ 12. Normalisation / 10 000 → Input 1 (N, 10, 36)
        ├─ 13. Extraction labels → (N,)
        ├─ 14. Vérification d'intégrité
        ├─ 15. Split Train/Val/Test (conforme Table 2 du paper)
        ├─ 16. Sauvegarde .npy (12 fichiers)
        └─ 17. Visualisations (taux manquants, NDVI, distributions)
                        │
                        ▼
FORMAT FINAL PRÊT POUR MCTNet
  Input 1 : (N, 10, 36) — réflectances normalisées [0, 1]
  Input 2 : (N, 36)     — masque binaire (1=valide, 0=manquant)
  Labels  : (N,)        — classe de culture (entier 0 à N_classes-1)
```

---

*Pipeline développé dans le cadre du projet de cartographie des cultures par apprentissage profond, basé sur Wang et al. (2024).*
