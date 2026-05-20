# Deep Learning for Crop Classification
## M1 SII — USTHB — 2025/2026

**Statut : ✅ Projet livré (rapport remis)**

Article de référence :
> Wang et al., 2024 — *"A lightweight CNN-Transformer network for pixel-based crop mapping using time-series Sentinel-2 imagery"*, Computers and Electronics in Agriculture

---

## Équipe

| Membre | Rôle principal |
|--------|---------------|
| Arslan | Partie 1 : ALPE + Transformer + assemblage MCTNet + train.py · Partie 3 : GatedMCTNet, MCTNetUSkip, UNetMCTNetWithCovars |
| Tesnime | Partie 1 : CNN sub-module · Partie 2 : MCTNetWithCovars · Partie 3 : MCTNet Multiscale |
| Sarah | Partie 1 : Data Preprocessing · Partie 2 : preprocessing covariables environnementales |

---

## Structure du projet

```
crop-classification/
├── README.md                              ← ce fichier
├── enonce_du_projet.md                    ← énoncé original du prof
├── avancement.md                          ← état d'avancement (figé, projet livré)
├── rapport/                               ← rapport LaTeX (source de vérité)
│   ├── main.tex
│   ├── chapters/{intro,partie1,partie2,partie3,conclusion}.tex
│   ├── figures/
│   └── references.bib
├── archive/                               ← rapport compilé + zip livré + brouillons
├── data/                                  ← données (non versionnées, voir plus bas)
│
├── Partie I — Reproduction MCTNet/        ← Partie 1 de l'énoncé
│   ├── resume.md
│   ├── Point 1 — Literature Review/
│   ├── Point 2 — Dataset Acquisition/
│   ├── Point 3 — Data Exploration/
│   ├── Point 4 — Data Preprocessing/
│   └── Point 5 — Model Implementation/
│       ├── src/         (transformer_alpe, cnn_submodule, ctfusion, mctnet)
│       ├── tests/
│       ├── docs/        (doc_*.md, point5.md)
│       ├── models/      (best_*.pth — non versionnés, voir Drive)
│       └── train.py
│
├── Partie II — Covariables Environnementales/   ← Partie 2 de l'énoncé
│   ├── README.md
│   ├── data/            (preprocessing_Part2.ipynb)
│   ├── src/             (MCTNetWithCovars.ipynb)
│   └── docs/            (Part2_Climat_Pipeline.md, covariables_par_culture.md, etc.)
│
└── Partie III — Contributions/            ← Partie 3 de l'énoncé : 3 contributions
    ├── README.md
    ├── gated/           ← Contribution 1 : GatedMCTNet (fusion dynamique)
    │   ├── notebooks/   (GatedMCTNet.ipynb — entraînement MCTNet vs GatedMCTNet)
    │   └── models/      (best_Arkansas_gated.pth, best_California_gated.pth)
    │   NOTE : code Python de GatedMCTNet dans Partie I/Point 5/src/{mctnet,ctfusion}.py
    ├── multiscale/      ← Contribution 2 : MCTNet Multiscale
    │   ├── src/         (CNN_MultiScale.py, ctfusion_MultiScale.py, ...)
    │   ├── notebooks/   (MCTNetMultiscale.ipynb)
    │   ├── docs/        (rapport_ctfusion_multiscale.md)
    │   └── results/     (figures confusion + courbes)
    ├── uskip/           ← Contribution 3 V1 : MCTNetUSkip (skip connections légers)
    │   └── notebooks/   (MCTNetUSkip.ipynb — auto-contenu)
    ├── unet/            ← Contribution 3 V2 : UNetMCTNetWithCovars (encodeur-décodeur + covariables)
    │   └── notebooks/   (UNetMCTNetWithCovars.ipynb)
    └── docs/            (idees_partie3.md — idées initiales)
```

---

## Données

Les fichiers de données ne sont pas versionnés dans ce repo (taille).
Ils sont disponibles sur le Google Drive partagé de l'équipe.

Structure attendue après téléchargement :

```
data/
└── preprocessed/
    ├── scale30/   (résolution 30m — version principale)
    │   ├── Arkansas_{train,val,test}_{input1,input2,labels}.npy
    │   └── California_{train,val,test}_{input1,input2,labels}.npy
    └── scale20/   (résolution 20m — comparaison)
        └── ...
```

---

## Comment relancer chaque partie

### Partie 1 — Reproduction MCTNet
```bash
cd "Partie I — Reproduction MCTNet/Point 5 — Model Implementation"
python train.py --region Arkansas --data_dir ../../data/preprocessed/scale30
python train.py --region California --data_dir ../../data/preprocessed/scale30
```

### Partie 2 — Covariables Environnementales
Notebooks Jupyter, exécution sur Colab recommandée (GPU). Voir `Partie II — Covariables Environnementales/README.md`.

### Partie 3 — Contributions
Notebooks Jupyter, exécution sur Colab. Voir `Partie III — Contributions/README.md`.

---

## Compilation du rapport

```bash
cd rapport/
pdflatex main.tex && biber main && pdflatex main.tex && pdflatex main.tex
```

Le PDF compilé livré est aussi disponible dans `archive/PFE (1).pdf`.

---

## Workflow Git

Le projet a été développé via des branches par membre :

```
main      ← stable
└── dev   ← intégration
    ├── arslan/transformer
    ├── tesnime/cnn
    ├── sarah/preprocessing  (sarah_code, sarah_data)
    └── refactor/structure   ← branche actuelle (réorganisation finale)
```

Pour le détail de l'avancement par point, voir `avancement.md`.
