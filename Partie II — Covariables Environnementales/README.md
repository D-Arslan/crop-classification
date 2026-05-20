# Partie II — Intégration des Covariables Environnementales

**Statut : ✅ Terminé**

## Objectif

Étendre MCTNet (Partie 1) avec une branche dédiée aux **covariables environnementales** (sol, climat, topographie) et mesurer leur apport via une étude d'ablation (7 configurations × 2 régions).

Cette partie correspond au **Chapitre 2 — Partie 2** du rapport ([rapport/chapters/partie2.tex](../rapport/chapters/partie2.tex)).

## Architecture — MCTNetWithCovars

- **Branche Sentinel-2** : identique à MCTNet de la Partie 1 (3 stages CTFusion + GMP → vecteur 80-dim)
- **Branche covariables** : MLP `Linear(K, 32) + ReLU + Dropout` où K ∈ {2, 3, 5, 108, 110, 111, 113} selon la config
- **Fusion** : concaténation 80 + 32 = 112, puis `Linear(112, N_classes)`

## Covariables intégrées (7 variables)

| Catégorie | Variable | Source | Format |
|---|---|---|---|
| **Sol** | pH (`ph_b0`) | OpenLandMap | pH réel ÷10 |
| **Sol** | Carbone Organique (`oc_b0`) | OpenLandMap | g/kg ÷5 |
| **Sol** | Texture (`texture_b0`) | OpenLandMap | Classe USDA 1-12 |
| **Topographie** | Élévation (`elevation`) | ETOPO1 (NOAA) | Mètres |
| **Topographie** | Modelé (`landforms`) | ALOS (CSP/ERGo) | Catégoriel |
| **Climat** | Température (`T_c`) | ERA5 Daily | °C |
| **Climat** | Précipitations (`P_mm`) | ERA5 Daily | mm |
| **Climat** | Humidité Relative (`RH`) | ERA5 Daily | % |

## Contenu du dossier

```
Partie II — Covariables Environnementales/
├── README.md                          ← ce fichier
├── data/
│   └── preprocessing_Part2.ipynb       ← pipeline de prétraitement covariables (extraction GEE, alignement, normalisation)
├── src/
│   └── MCTNetWithCovars.ipynb          ← entraînement MCTNet + branche covariables (7 configs × 2 régions = 14 runs)
└── docs/
    ├── Part2_Climat_Pipeline.md        ← documentation pipeline climat
    ├── covariables_par_culture.md      ← analyse exploratoire covariables / classes
    ├── rapport_dataset_MCTNet.md       ← documentation du dataset enrichi
    └── rapport_code_acquisition_preprocessing.md
```

## Comment relancer

Notebook Jupyter, exécution Colab recommandée. Les fichiers `.npy` enrichis avec covariables sont attendus dans `data/preprocessed/scaleXX/` (généré par `data/preprocessing_Part2.ipynb`).

## Résultats

Voir le [chapitre Partie 2 du rapport](../rapport/chapters/partie2.tex) — étude d'ablation complète, tableaux et figures dans `rapport/figures/California_*_{f1,oa,kappa,loss}.png`.

## Référence interne

⚠️ Le notebook `Partie III — Contributions/unet/notebooks/UNetMCTNetWithCovars.ipynb` (Partie 3 V2) réutilise les données préparées par cette Partie 2. La Partie 3 (U-Net + covariables) dépend donc des `.npy` générés ici.
