# Point 1 — Literature Review

## Statut : ✅ Terminé

## Objectif
Lire et analyser en détail l'article de référence pour comprendre l'architecture MCTNet, les choix de design, et les résultats à reproduire.

## Fichiers produits
- `rapport_literature_review.docx` — rapport complet
- `1-Lightweight-CNN-Transformer_Crop_Mapping.pdf` — article original

---

## Détails techniques

### Architecture MCTNet — lecture de l'article

#### Entrées du modèle
- **Input 1** : matrice `10 × 36` (bandes spectrales × timesteps)
- **Input 2** : vecteur `1 × 36` (masque de données manquantes, 0=manquant, 1=valide)
- Les valeurs 0 dans Input 1 signalent les timesteps manquants (nuages, absence d'image)

#### Bloc CTFusion (Section 2.2)
Unité de base répétée 3 fois. À chaque stage :
- Le CNN extrait des features locales (temporelles)
- Le Transformer capture les dépendances globales
- Une fusion combine les deux
- Un stride=2 réduit la dimension temporelle de moitié à chaque stage

| Stage | Entrée | Sortie |
|-------|--------|--------|
| 1 | (10, 36) | (20, 18) |
| 2 | (20, 18) | (40, 9) |
| 3 | (40, 9) | (80, 1) |

Après les 3 stages : Global Max Pooling → vecteur (80,) → MLP → N classes

#### CNN sub-module (Section 2.2)
- 2 couches Conv1D sur la dimension temporelle
- BatchNorm après chaque couche
- ReLU après les 2 couches
- Connexion résiduelle : sortie = entrée + conv(entrée)

#### Transformer sub-module (Section 2.3)
- Basé sur l'encodeur du Transformer original (Vaswani et al., 2017)
- Multi-Head Self-Attention (n_head=5)
- Feed Forward Network
- Add & Norm après chaque bloc
- L'ALPE remplace l'encodage positionnel standard **uniquement dans le Stage 1**

#### ALPE — Adaptive Learnable Positional Encoding (Section 2.4)
Formule : `ALPE(t) = ECA(Conv1D(PE(t) × mask))`
1. PE sinusoïdal classique sur les timesteps
2. Multiplication par le masque → annule les positions manquantes
3. Conv1D → capture les relations temporelles locales
4. ECA (Efficient Channel Attention) → pondération inter-canaux
5. Addition résiduelle à x

**Motivation** : l'encodage positionnel standard attribue une position à tous les timesteps, même manquants. L'ALPE adapte l'encodage en ignorant les positions sans données.

### Hyperparamètres (Table 3)
| Paramètre | Valeur |
|-----------|--------|
| n_stage | 3 |
| n_head | 5 |
| kernel_size | 3 |
| optimizer | Adam |
| lr | 0.001 |
| batch_size | 32 |
| epochs | 200 |

### Données utilisées dans l'article
- Sentinel-2, 10 bandes, année 2021
- Composites médianes sur des fenêtres de 10 jours → 36 timesteps
- Deux zones : Arkansas (5 classes) et Californie (6 classes)
- Split : 300 échantillons/classe → 240 train + 60 val, reste en test

### Résultats de l'article (Table 5)
| Zone | OA | Kappa | F1 |
|------|-----|-------|-----|
| Arkansas | 0.968 | 0.951 | 0.933 |
| Californie | 0.852 | 0.806 | 0.829 |

### Figures clés
- **Figure 1** : schéma global de l'architecture MCTNet
- **Figure 2** : courbes NDVI moyennes par classe (référence pour valider notre exploration)
- **Figure 3** : détail du bloc CTFusion
- **Figure 4** : détail de l'ALPE

---

## Problèmes rencontrés & solutions
Aucun — point purement documentaire.

---

## Décisions prises & pourquoi
- Lecture exhaustive de toutes les sections, y compris annexes et ablation study
- L'ablation study (Table 4) confirme que l'ALPE apporte un gain significatif par rapport à un PE standard → justifie l'effort d'implémentation

---

## Lien avec les autres points
- → **Point 2** : les paramètres de collecte (10 bandes, 36 timesteps, année 2021) sont dictés par cet article
- → **Point 5** : toutes les décisions d'architecture s'appuient sur les sections 2.2, 2.3, 2.4 et Table 3
