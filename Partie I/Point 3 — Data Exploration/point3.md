# Point 3 — Data Exploration

## Statut : ✅ Terminé

## Objectif
Explorer les CSV produits en Point 2 pour valider leur qualité, visualiser les profils spectraux et NDVI, analyser les données manquantes, et s'assurer que les données sont cohérentes avec l'article avant de passer au preprocessing.

## Fichiers produits
- `03_exploration_donnees_v2.ipynb` — notebook final (version corrigée)
- `rapport_point3_v2_complet.docx` — rapport complet
- `output_V2/ndvi_timeseries_2.png` — courbes NDVI par classe
- `output_V2/spectral_profiles_2.png` — profils spectraux moyens
- `output_V2/missing_data_2.png` — carte des données manquantes par timestep
- `output_V2/missing_per_sample_2.png` — distribution du % de manquants par pixel
- `V1_avant_correction/` — version initiale (conservée pour traçabilité)

---

## Détails techniques

### Structure du notebook v2
1. Chargement des CSV et aperçu
2. Analyse des données manquantes (taux de zéros par bande et par timestep)
3. Courbes NDVI moyennes par classe
4. Profils spectraux moyens par classe
5. Matrice de corrélation inter-bandes

### Calcul du NDVI
```python
nir = data[:, band_idx['B8'], :]   # bande NIR
red = data[:, band_idx['B4'], :]   # bande Red
denominator = nir + red
ndvi = np.where(denominator > 0, (nir - red) / denominator, np.nan)
# np.nan sur les pixels où NIR+RED=0 (données manquantes)
mean_ndvi = np.nanmean(ndvi[mask], axis=0)  # nanmean ignore les NaN
```

### Résultats Arkansas
| Métrique | Valeur |
|----------|--------|
| Taux de zéros global | 23.9% |
| Timesteps 100% manquants | 3 (t3, t15, t34) |
| Max spectral | 12 998 |
| Moyenne spectrale | 1 612 |

Décalage phénologique observé (cohérent avec l'article) :
- Maïs : pic NDVI DOY ~170
- Riz : pic NDVI DOY ~200
- Soja : pic NDVI DOY ~220
- Coton : pic NDVI DOY ~240

### Résultats Californie
| Métrique | Valeur |
|----------|--------|
| Taux de zéros global | 16.1% |
| Timesteps 100% manquants | 0 |
| Max spectral | 9 200 |
| Moyenne spectrale | 1 647 |

### Validation par rapport à l'article
- Courbes NDVI cohérentes avec la Figure 2 de l'article ✅
- Profils spectraux : pic NIR caractéristique de la végétation ✅
- Corrélations négatives visible/NIR (principe physique du NDVI) ✅
- Taux de données manquantes à des niveaux normaux pour Sentinel-2 ✅

---

## Problèmes rencontrés & solutions

### Problème v1 : artefacts dans les courbes NDVI
**Symptôme** : chutes brusques à 0 sur les courbes NDVI à certains timesteps.

**Cause** : les valeurs 0 (données manquantes) étaient traitées comme des réflectances réelles et incluses dans les moyennes.

```python
# v1 (incorrect) — les 0 faussent la moyenne
ndvi = np.where(denominator > 0, (nir - red) / denominator, 0)
mean_ndvi = np.mean(ndvi[mask], axis=0)
```

**Solution v2** : remplacer 0 par NaN avant les calculs, utiliser `nanmean`.

```python
# v2 (correct)
ndvi = np.where(denominator > 0, (nir - red) / denominator, np.nan)
mean_ndvi = np.nanmean(ndvi[mask], axis=0)
```

**Important** : cette correction s'applique **uniquement aux graphiques**. Les CSV ne sont pas modifiés. Les 0 restent dans les CSV car le modèle MCTNet en a besoin pour construire le masque (Input 2).

---

## Décisions prises & pourquoi
| Décision | Raison |
|----------|--------|
| Ne pas modifier les CSV | Les 0 sont une convention de l'article, utilisée par l'ALPE |
| Remplacer 0 par NaN uniquement pour les graphiques | Éviter les artefacts visuels sans altérer les données |
| Conserver la v1 dans un sous-dossier | Traçabilité de la correction |

---

## Lien avec les autres points
- ← **Point 2** : explore les CSV produits par la collecte GEE
- → **Point 4** : valide que les données sont exploitables ; confirme que les 0 sont des manquants à traiter
- → **Point 5** : les 3 timesteps 100% manquants en Arkansas (t3, t15, t34) sont un cas limite à garder en tête lors des tests du modèle
