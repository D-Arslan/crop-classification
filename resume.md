# RÉSUMÉ DU PROJET — Deep Learning for Crop Classification
## M1 SII — USTHB — 2025/2026

---

## 0. Navigation rapide

| Section | Contenu |
|---------|---------|
| [1. Contexte & Équipe](#1-contexte--équipe) | Article, rôles, workflow |
| [2. Données](#2-données) | Structure CSV, classes, statistiques |
| [3. Point 1 — Literature Review](#3-point-1--literature-review) | Résumé, livrables |
| [4. Point 2 — Dataset Acquisition](#4-point-2--dataset-acquisition) | GEE, zones, scripts |
| [5. Point 3 — Data Exploration](#5-point-3--data-exploration) | Notebook, résultats |
| [6. Point 4 — Data Preprocessing](#6-point-4--data-preprocessing) | Specs d'interface |
| [7. Point 5 — Model Implementation](#7-point-5--model-implementation) | Architecture, ALPE + Transformer |
| [8. Divergences avec l'article](#8-divergences-avec-larticle) | Écarts documentés |
| [9. Prochaines étapes](#9-prochaines-étapes) | Todo par personne |

---

## 1. Contexte & Équipe

### Article de référence
- **Titre** : "A lightweight CNN-Transformer network for pixel-based crop mapping using time-series Sentinel-2 imagery"
- **Auteurs** : Wang et al., 2024
- **Journal** : Computers and Electronics in Agriculture
- **Modèle** : MCTNet (Multi-stage CNN-Transformer Network)
- **PDF** : `Partie I/Point 1 — Literature Review/1-Lightweight-CNN-Transformer_Crop_Mapping.pdf`

### Rôles dans l'équipe

| Membre | Rôle |
|--------|------|
| Arslan Point 5 — ALPE + Transformer sub-module |
| Tesnime | Point 5 — CNN sub-module |
| Sarah | Point 4 — Data Preprocessing |

### État d'avancement global

| Point | Intitulé | Statut |
|-------|----------|--------|
| 1 | Literature Review | ✅ Terminé |
| 2 | Dataset Acquisition | ✅ Terminé |
| 3 | Data Exploration | ✅ Terminé |
| 4 | Data Preprocessing | ⏳ En cours (Sarah) |
| 5 | Model Implementation | ⏳ En cours |

---

## 2. Données

### Fichiers CSV finaux
- `Partie I/Point 2 — Dataset Acquisition/Arkansas_10k.csv` — 10 000 lignes × 361 colonnes
- `Partie I/Point 2 — Dataset Acquisition/California_10k.csv` — 10 000 lignes × 361 colonnes

### Structure des CSV
- Chaque ligne = 1 pixel agricole
- 360 colonnes spectrales = 10 bandes Sentinel-2 × 36 timesteps (composites médianes de 10 jours, année 2021)
- Nommage GEE : `B8` (t=0), `B8_1` (t=1), ..., `B8_35` (t=35)
- Colonne `label` = classe de culture (entier)
- Valeurs à 0 = données manquantes (nuages/absence d'image) — convention de l'article, **ne pas modifier**

### Bandes Sentinel-2 (10)
B2 (Blue), B3 (Green), B4 (Red), B5 (Red Edge 1), B6 (Red Edge 2), B7 (Red Edge 3), B8 (NIR), B8A (Red Edge 4), B11 (SWIR 1), B12 (SWIR 2)

### Classes Arkansas (5)
| Label | Nom | Points | % |
|-------|-----|--------|---|
| 0 | Corn | 1 773 | 17.7% |
| 1 | Cotton | 762 | 7.6% |
| 2 | Rice | 2 787 | 27.9% |
| 3 | Soybeans | 4 506 | 45.1% |
| 4 | Others | 172 | 1.7% |

### Classes Californie (6)
| Label | Nom | Points | % |
|-------|-----|--------|---|
| 0 | Rice | 3 653 | 36.5% |
| 1 | Alfalfa | 1 288 | 12.9% |
| 2 | Grapes | 1 580 | 15.8% |
| 3 | Almonds | 1 054 | 10.5% |
| 4 | Pistachios | 751 | 7.5% |
| 5 | Others | 1 674 | 16.7% |

### Statistiques
| Métrique | Arkansas | Californie |
|----------|----------|------------|
| Taux de zéros | 23.9% | 16.1% |
| Timesteps 100% manquants | 3 (t3, t15, t34) | 0 |
| Max spectral | 12 998 | 9 200 |
| Moyenne spectrale | 1 612 | 1 647 |

---

## 3. Point 1 — Literature Review

**Statut : ✅ Terminé**

### Livrable
- `Partie I/Point 1 — Literature Review/rapport_literature_review.docx`

### Points clés retenus de l'article
- MCTNet combine CNN (local) + Transformer (global) dans 3 stages successifs
- L'ALPE gère les données manquantes dès l'encodage positionnel
- Résultats cibles : OA=0.968 / Kappa=0.951 / F1=0.933 (Arkansas)

---

## 4. Point 2 — Dataset Acquisition

**Statut : ✅ Terminé**

### Livrables
- `Partie I/Point 2 — Dataset Acquisition/gee_arkansas_final4.js`
- `Partie I/Point 2 — Dataset Acquisition/gee_california_final4.js`
- `Partie I/Point 2 — Dataset Acquisition/gee_california_fina4_extra.js`
- `Partie I/Point 2 — Dataset Acquisition/merge_and_subsample.py`
- `Partie I/Point 2 — Dataset Acquisition/rapport_collecte_donnees.docx`

### Stratégie de collecte GEE
Impossible d'échantillonner sur l'état entier (timeout GEE après 21-35 min). Solution : découpage en petites bounding boxes (~0.5° × 0.4°) sur les zones agricoles, export par zone, fusion en Python, sous-échantillonnage à 10 000 pixels.

### Zones Arkansas (5)
| Zone | Région | Bounding Box |
|------|--------|-------------|
| Z1 | Delta central | [-91.75, 34.40, -91.25, 34.80] |
| Z2 | Sud Delta | [-92.10, 33.80, -91.60, 34.20] |
| Z3 | Stuttgart (riz) | [-91.65, 34.60, -91.15, 35.00] |
| Z4 | Jonesboro (maïs/soja) | [-90.90, 35.60, -90.40, 36.00] |
| Z5 | Pine Bluff (coton/soja) | [-91.80, 33.40, -91.30, 33.80] |

### Zones Californie (8)
| Zone | Région | Bounding Box |
|------|--------|-------------|
| Z1 | Sacramento Valley | [-122.20, 39.00, -121.80, 39.40] |
| Z2 | San Joaquin (centre) | [-120.10, 36.80, -119.70, 37.20] |
| Z3 | Fresno (raisins/amandes) | [-119.95, 36.55, -119.55, 36.95] |
| Z4 | Bakersfield (pistaches) | [-119.40, 35.20, -119.00, 35.60] |
| Z5 | Colusa (riz) | [-122.10, 39.10, -121.70, 39.50] |
| Z6 | Imperial Valley (alfalfa) | [-115.70, 32.70, -115.30, 33.10] |
| Z7 | Sud Kern County | [-119.70, 35.40, -119.30, 35.80] |
| Z8 | Tulare County | [-119.50, 35.90, -119.10, 36.30] |

### Masques appliqués
1. CDL agricole : codes 1-61 (Arkansas) + 69, 75, 204 (Californie perennials)
2. CDL confiance ≥ 95%
3. ESA WorldCover v200 2021 : classe 40 (Cropland)
4. Masque nuages QA60 : bits 10 et 11
5. Filtre CLOUDY_PIXEL_PERCENTAGE < 20%

### Paramètres d'échantillonnage
- `scale: 30` (résolution CDL)
- `numPixels: 10000`, `seed: 42`, `tileScale: 16`
- Méthode : `sample()` (stratifiedSample causait des timeouts)

### Astuce technique clé : safetyImage
`ee.Algorithms.If` ne fonctionnait pas dans notre environnement GEE. Solution :
```javascript
var safetyImage = ee.Image.constant([0,0,0,0,0,0,0,0,0,0])
  .rename(bands).toFloat()
  .set('system:time_start', ee.Date('2021-01-01').millis())
  .set('CLOUDY_PIXEL_PERCENTAGE', 0)
  .selfMask();

var safeColl = collection
  .map(function(img) { return img.toFloat(); })
  .merge(ee.ImageCollection([safetyImage]));
return safeColl.median().unmask(0).toFloat();
```

---

## 5. Point 3 — Data Exploration

**Statut : ✅ Terminé**

### Livrables
- `Partie I/Point 3 — Data Exploration/03_exploration_donnees_v2.ipynb`
- `Partie I/Point 3 — Data Exploration/rapport_point3_v2_complet.docx`
- Figures dans `Partie I/Point 3 — Data Exploration/output_V2/`

### Correction v1 → v2
**Problème** : les valeurs 0 (données manquantes) faussaient les moyennes NDVI et corrélations.
**Solution** : remplacer 0 par NaN uniquement pour les calculs graphiques.

```python
# v2 (correct) — uniquement pour les graphiques, les CSV ne sont pas modifiés
ndvi = np.where(denominator > 0, (nir - red) / denominator, np.nan)
mean_ndvi = np.nanmean(ndvi[mask], axis=0)
```

### Résultats validés
- Courbes NDVI cohérentes avec la Figure 2 de l'article
- Décalage phénologique : maïs (DOY 170) → riz (DOY 200) → soja (DOY 220) → coton (DOY 240)
- Données manquantes à des niveaux normaux (23.9% Arkansas, 16.1% Californie)
- Corrélations négatives visible/NIR physiquement correctes (principe NDVI)

---

## 6. Point 4 — Data Preprocessing

**Statut : ⏳ En cours (Coéquipier B)**

### Specs d'interface — ce que ce module doit produire

Le module de preprocessing reçoit les CSV bruts et doit fournir au modèle (Point 5) des tenseurs dans le format suivant :

#### Tenseur d'entrée X — Input 1
- **Forme** : `(B, 10, 36)` — batch × bandes × timesteps
- **Type** : `torch.float32`
- **Ordre des bandes** : B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 (index 0 à 9)
- **Normalisation** : à définir par Coéquipier B ← *compléter ici*

#### Tenseur masque M — Input 2
- **Forme** : `(B, 36)` — batch × timesteps
- **Type** : `torch.float32`
- **Valeurs** : `1.0` = timestep valide, `0.0` = timestep manquant (valeur originale = 0)
- **Construction** : `mask = (X.sum(dim=1) != 0).float()` — un timestep est manquant si toutes ses bandes sont à 0

#### Labels Y
- **Forme** : `(B,)`
- **Type** : `torch.long` (requis par `nn.CrossEntropyLoss`)
- **Valeurs** : entiers de 0 à N_classes-1 (inchangés par rapport aux CSV)

#### Split train/val/test (Table 2 de l'article)
- 300 échantillons par classe → 240 train + 60 validation
- Le reste → test
- Arkansas : ~1200 train + ~300 val + ~8500 test
- Californie : ~1440 train + ~360 val + ~8200 test

---

## 7. Point 5 — Model Implementation

**Statut : ⏳ En cours**

### Architecture MCTNet — vue d'ensemble
```
Input 1 : (B, 10, 36)   ← matrice bandes × timesteps
Input 2 : (B, 36)       ← masque de manquants

CTFusion Stage 1 : CNN(10×36) + Transformer/ALPE(10×36) → fusion → (20×18)
CTFusion Stage 2 : CNN(20×18) + Transformer(20×18)      → fusion → (40×9)
CTFusion Stage 3 : CNN(40×9)  + Transformer(40×9)       → fusion → (80×1)

Global Max Pooling → (B, 80)
MLP Classifier    → (B, N_classes)
```

### Hyperparamètres (Table 3 de l'article)
| Paramètre | Valeur |
|-----------|--------|
| n_stage | 3 |
| n_head | 5 |
| kernel_size | 3 |
| optimizer | Adam |
| lr | 0.001 |
| batch_size | 32 |
| epochs | 200 |

### Métriques d'évaluation
- Overall Accuracy (OA)
- Kappa Coefficient
- Macro-averaged F1 Score

### Résultats cibles (Table 5 de l'article)
| Zone | OA | Kappa | F1 |
|------|-----|-------|-----|
| Arkansas | 0.968 | 0.951 | 0.933 |
| Californie | 0.852 | 0.806 | 0.829 |

---

### 7a. ALPE + Transformer sub-module ← [Moi]

**Statut : ⏳ En cours — module terminé (v2), intégration CTFusion à faire**

#### Fichier
- `Partie I/Point 5 — Model Implementation/transformer_alpe.py`

#### Modules implémentés

**`sinusoidal_pe(seq_len, d_model)`**
- Encodage positionnel sinusoïdal classique (Vaswani et al., 2017)
- Retourne `(T, C)`

**`ECA(channels)`** — Efficient Channel Attention
- Global Average Pooling → Conv1D locale (kernel adaptatif `~log2(C)`) → Sigmoid
- Entrée/sortie : `(B, C, T)` — pondération inter-canaux sans réduction de dimension

**`ALPE(channels, seq_len, kernel_size=3)`** — Adaptive Learnable Positional Encoding
- Implémente : `ALPE(t) = ECA(Conv1D(PE(t) ⊙ mask))`
- Étapes : PE sinusoïdal → masquage (×0 sur timesteps manquants) → Conv1D + BN + ReLU → ECA → addition résiduelle à x
- Utilisé **uniquement dans le Stage 1** (seul stage avec masque disponible en entrée)
- Entrée : `(B, C, T)` + mask `(B, T)` — Sortie : `(B, C, T)`

**`TransformerSubModule(channels, seq_len, n_head=5, use_alpe=False, kernel_size=3)`**
- ALPE optionnel (activé si `use_alpe=True`) → Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm
- FFN : `Linear(C, 4C) → ReLU → Dropout → Linear(4C, C)`
- Format entrée/sortie : `(B, C, T)` — compatible avec le CNN sub-module du coéquipier

#### Décisions d'implémentation
| Décision | Raison |
|----------|--------|
| `batch_first=True` dans MultiheadAttention | Cohérence avec le format `(B, C, T)` |
| FFN dim cachée = 4×C | Convention Transformer standard (l'article ne précise pas) |
| BN après Conv1D dans ALPE | Stabilité d'entraînement |
| Addition résiduelle dans ALPE | L'article formule ALPE comme un ajout à x |

#### Résultats des tests
| Stage | Input | Mask | Output | Params |
|-------|-------|------|--------|--------|
| 1 (avec ALPE) | `(B, 10, 36)` | `(B, 36)` | `(B, 10, 36)` | — |
| 2 | `(B, 20, 18)` | — | `(B, 20, 18)` | — |
| 3 | `(B, 40, 9)` | — | `(B, 40, 9)` | — |
| **Total 3 Transformers** | | | | **26 433** |

#### Ce qui reste à faire
- 🔲 Intégration dans le bloc CTFusion (avec le CNN du coéquipier)
- 🔲 Implémentation de la fusion CNN + Transformer (CTFusion)
- 🔲 Tests sur données réelles (Arkansas_10k.csv / California_10k.csv)

---

### 7b. CNN sub-module ← Coéquipier A

**Statut : ⏳ En cours**

#### Architecture attendue
- 2 couches Conv1D (dimension temporelle)
- BatchNorm après chaque couche
- Activation ReLU après les 2 couches
- Connexion résiduelle : output = input + conv(input)
- Format entrée/sortie : `(B, C, T)` — même convention que le Transformer

---

## 8. Divergences avec l'article

| # | Divergence | Impact estimé |
|---|-----------|---------------|
| 1 | Échantillonnage par zones (~0.5°×0.4°) au lieu de l'état entier | Distribution spatiale moins représentative |
| 2 | Classe Others sous-représentée (1.7% vs 6.2% article) | F1 Others potentiellement plus bas |
| 3 | Rice surreprésenté en Californie (36.5% vs 20.4%) | Biais vers Rice en Californie |
| 4 | Scale=30 — l'article ne précise pas | Probablement sans impact |
| 5 | Courbes NDVI plus bruitées — variabilité locale des petites zones | Cosmétique, sans impact sur le modèle |
| 6 | `ee.Algorithms.If` non fonctionnel — contourné avec safetyImage | Aucun (comportement identique) |
| 7 | `stratifiedSample` causait des timeouts — remplacé par `sample` | Distribution des classes légèrement différente |

---

## 9. Prochaines étapes

### Arslan — ALPE + Transformer
- 🔲 Implémenter le bloc CTFusion (fusion CNN + Transformer) une fois le CNN disponible
- 🔲 Tester l'intégration sur un batch synthétique puis sur données réelles

### Tesnime — CNN sub-module
- ⏳ Finaliser l'implémentation CNN

### Sarah — Data Preprocessing
- ⏳ Implémenter le pipeline preprocessing
- ⏳ Compléter la section normalisation dans [Section 6](#6-point-4--data-preprocessing)

### Ensemble
- 🔲 Intégration complète MCTNet (CTFusion × 3 + GMP + MLP)
- 🔲 Entraînement sur Arkansas et Californie
- 🔲 Évaluation OA / Kappa / F1 et comparaison avec Table 5 de l'article
