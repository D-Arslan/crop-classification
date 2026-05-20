# Partie III — Contributions

**Statut : ✅ Terminé**

## Objectif

Proposer et évaluer des extensions originales de MCTNet inspirées de la littérature.
Trois contributions indépendantes ont été développées, chacune ciblant un aspect différent de l'architecture.

Cette partie correspond au **Chapitre 3 — Partie 3** du rapport ([rapport/chapters/partie3.tex](../rapport/chapters/partie3.tex)).

## Vue d'ensemble des contributions

| # | Contribution | Cible | Sous-dossier |
|---|---|---|---|
| 1 | **GatedMCTNet** — fusion dynamique CNN/Transformer apprise (gate) | Fusion des branches | `gated/` |
| 2 | **MCTNet Multiscale** — CTFusion à champs réceptifs multiples | Résolution temporelle / contexte local | `multiscale/` |
| 3 V1 | **MCTNetUSkip** — connexions skip légères vers le classificateur | Préservation des représentations intermédiaires | `uskip/` |
| 3 V2 | **UNetMCTNetWithCovars** — encodeur-décodeur U-Net + covariables environnementales | Représentation hiérarchique + données auxiliaires | `unet/` |

> Les variantes V1 et V2 de la **Contribution 3** sont distinctes mais partagent la même motivation
> (réintroduire les détails phénologiques perdus par les MaxPool successifs).
> V1 est une version légère sans covariables (sur données Partie 1).
> V2 est une architecture U-Net complète avec branche covariables (sur données Partie 2).

## Structure du dossier

```
Partie III — Contributions/
├── README.md                                     ← ce fichier
├── docs/
│   └── idees_partie3.md                          ← idées initiales explorées en début de Partie 3
│
├── gated/                                        ← Contribution 1 : GatedMCTNet
│   ├── notebooks/
│   │   ├── GatedMCTNet.ipynb                     ← architecture + entraînement
│   │   └── train_colab.ipynb                     ← script utilitaire Colab
│   └── models/                                   ← best_*.pth (non versionnés, voir Drive)
│
├── multiscale/                                   ← Contribution 2 : MCTNet Multiscale
│   ├── src/
│   │   ├── CNN_MultiScale.py                     ← CNN multi-receptive-field
│   │   ├── ctfusion_MultiScale.py                ← bloc CTFusion adapté
│   │   ├── mctnet.py                             ← assemblage
│   │   ├── transformer_alpe.py                   ← Transformer + ALPE
│   │   └── train.py
│   ├── notebooks/
│   │   └── MCTNetMultiscale.ipynb
│   ├── docs/
│   │   └── rapport_ctfusion_multiscale.md
│   └── results/
│       ├── confusion_Arkansas.png
│       ├── confusion_California_scale.png
│       ├── curves_Arkansas.png
│       └── curves_California.png
│
├── uskip/                                        ← Contribution 3 V1 : MCTNetUSkip
│   └── notebooks/
│       └── MCTNetUSkip.ipynb                     ← À AJOUTER (notebook local, non pushé)
│
└── unet/                                         ← Contribution 3 V2 : UNetMCTNetWithCovars
    └── notebooks/
        └── UNetMCTNetWithCovars.ipynb            ← U-Net complet + branche covariables (utilise les .npy de la Partie 2)
```

## Comment relancer

Notebooks Jupyter, exécution Colab recommandée (GPU).

| Contribution | Données nécessaires |
|---|---|
| GatedMCTNet | `data/preprocessed/scale30/` (Partie 1) |
| Multiscale | `data/preprocessed/scale30/` (Partie 1) |
| MCTNetUSkip | `data/preprocessed/scale30/` (Partie 1) |
| UNetMCTNetWithCovars | `.npy` enrichis avec covariables (générés par la Partie 2) |

## Résultats

Voir le [chapitre Partie 3 du rapport](../rapport/chapters/partie3.tex) :
- GatedMCTNet : courbes dans `rapport/figures/courbes_gated.png`
- Multiscale : `multiscale/results/` + figures `rapport/figures/`
- MCTNetUSkip : tableaux comparatifs dans le rapport
- UNetMCTNetWithCovars : étude d'ablation par groupes de covariables, figures `rapport/figures/California_*_{f1,oa,kappa,loss}.png` et `rapport/figures/confusion_comparison_*.png`
