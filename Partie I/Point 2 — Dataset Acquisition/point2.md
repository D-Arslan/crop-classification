# Point 2 — Dataset Acquisition

## Statut : ✅ Terminé

## Objectif
Collecter via Google Earth Engine les données Sentinel-2 pour Arkansas et Californie, produire deux CSV de 10 000 pixels avec 360 bandes spectrales + label.

## Fichiers produits
- `gee_arkansas_final4.js` — script GEE pour les 5 zones Arkansas
- `gee_california_final4.js` — script GEE pour les 5 zones Californie (Z1-Z5)
- `gee_california_fina4_extra.js` — script GEE pour les 3 zones supplémentaires (Z6-Z8)
- `merge_and_subsample.py` — fusion des zones et sous-échantillonnage à 10 000
- `rapport_collecte_donnees.docx` — rapport complet
- `Arkansas_10k.csv` — données finales Arkansas
- `California_10k.csv` — données finales Californie
- `Zones utilisées pour Arkansas/` — CSVs par zone avant fusion
- `Zones utilisées pour Californie/` — CSVs par zone avant fusion

---

## Détails techniques

### Paramètres de collecte
| Paramètre | Valeur | Raison |
|-----------|--------|--------|
| Satellite | Sentinel-2 L2A | Même que l'article |
| Bandes | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 | 10 bandes spectrales de l'article |
| Période | Année 2021 | Même que l'article |
| Fenêtre temporelle | 10 jours | Composites médianes de 10 jours |
| Timesteps | 36 | 365 jours / 10 jours ≈ 36 |
| Scale | 30 m | Résolution du CDL (l'article ne précise pas) |
| numPixels | 10 000 | Taille finale souhaitée |
| seed | 42 | Reproductibilité |
| tileScale | 16 | Éviter les memory limit errors GEE |

### Stratégie de zonage
L'échantillonnage sur l'état entier provoquait des timeouts GEE (21-35 min). Solution : découper chaque état en petites bounding boxes (~0.5° × 0.4°) ciblant les zones agricoles connues, exporter chaque zone séparément, fusionner en Python.

**Arkansas : 5 zones**
| Zone | Région | Bounding Box |
|------|--------|-------------|
| Z1 | Delta central | [-91.75, 34.40, -91.25, 34.80] |
| Z2 | Sud Delta | [-92.10, 33.80, -91.60, 34.20] |
| Z3 | Stuttgart (riz) | [-91.65, 34.60, -91.15, 35.00] |
| Z4 | Jonesboro (maïs/soja) | [-90.90, 35.60, -90.40, 36.00] |
| Z5 | Pine Bluff (coton/soja) | [-91.80, 33.40, -91.30, 33.80] |

**Californie : 8 zones** (5 initiales + 3 ajoutées pour alfalfa/pistaches sous-représentés)
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

### Masques appliqués (dans l'ordre)
1. **CDL agricole** — codes 1-61 (cultures annuelles) + 69, 75, 204 (Californie : alfalfa, pistaches, almonds)
2. **CDL confiance ≥ 95%** — filtre les pixels CDL incertains
3. **ESA WorldCover v200 2021, classe 40** (Cropland) — double validation
4. **Masque nuages QA60** — bits 10 (opaque clouds) et 11 (cirrus)
5. **CLOUDY_PIXEL_PERCENTAGE < 20%** — filtre au niveau de l'image entière

### Script Python de fusion
```python
ark = pd.concat([pd.read_csv(f) for f in ark_files], ignore_index=True)
ark = ark.drop(columns=['system:index', '.geo'], errors='ignore')
ark_sampled = ark.sample(n=10000, random_state=42).reset_index(drop=True)
ark_sampled.to_csv('Arkansas_10k.csv', index=False)
```

### Structure des CSV produits
- 10 000 lignes × 361 colonnes
- 360 colonnes spectrales : nommage GEE `B2`, `B2_1`, ..., `B2_35`, `B3`, `B3_1`, ...
- 1 colonne `label` (entier, classe de culture)
- Valeurs 0 = données manquantes (pas de NaN, pas de négatifs)

---

## Problèmes rencontrés & solutions

| Problème | Solution |
|----------|----------|
| Timeout GEE sur l'état entier | Découpage en petites bounding boxes par zone agricole |
| `ee.Algorithms.If` non fonctionnel | Remplacement par une `safetyImage` masquée (voir ci-dessous) |
| `stratifiedSample` causait des timeouts | Remplacement par `sample()` + sous-échantillonnage Python |
| Alfalfa et pistaches sous-représentés en Californie | Ajout de 3 zones supplémentaires (Z6, Z7, Z8) |
| Colonnes `system:index` et `.geo` inutiles dans l'export | Supprimées dans `merge_and_subsample.py` |

### Détail : astuce safetyImage
`ee.Algorithms.If` était censé retourner une image vide si la collection Sentinel-2 était vide pour un timestep donné, mais ne fonctionnait pas dans notre environnement. Solution :

```javascript
var safetyImage = ee.Image.constant([0,0,0,0,0,0,0,0,0,0])
  .rename(bands).toFloat()
  .set('system:time_start', ee.Date('2021-01-01').millis())
  .set('CLOUDY_PIXEL_PERCENTAGE', 0)
  .selfMask(); // masquée partout — n'affecte pas median()

var safeColl = collection
  .map(function(img) { return img.toFloat(); })
  .merge(ee.ImageCollection([safetyImage]));
return safeColl.median().unmask(0).toFloat();
// unmask(0) : les pixels sans données valides reviennent à 0
```
La safetyImage étant masquée partout, elle ne contribue jamais à la médiane mais garantit que la collection n'est jamais vide (ce qui ferait planter GEE).

---

## Décisions prises & pourquoi
| Décision | Raison |
|----------|--------|
| `scale=30` (résolution CDL) | L'article ne précise pas ; le CDL est à 30 m, c'est cohérent |
| Valeurs 0 pour les manquants (pas NaN) | Convention explicite de l'article — utilisée par l'ALPE pour construire le masque |
| Sous-échantillonnage aléatoire (pas stratifié) | `stratifiedSample` causait des timeouts GEE |

---

## Lien avec les autres points
- → **Point 3** : les CSV produits ici sont analysés dans l'exploration
- → **Point 4** : le preprocessing part de ces CSV ; les 0 ne sont pas des NaN, ils signalent les manquants
- → **Point 5** : l'Input 2 (masque) est construit à partir des 0 dans ces CSV
