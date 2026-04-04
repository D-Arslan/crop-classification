# Point 4 — Data Preprocessing

## Statut : ⏳ En cours

## Objectif
Transformer les CSV bruts (Arkansas_10k.csv, California_10k.csv) en tenseurs PyTorch prêts à être consommés par MCTNet. Produire un DataLoader utilisable directement par le Point 5.

## Fichiers produits
- Arkansas_train_input1.npy  — shape (1200, 10, 36)    — features train
- Arkansas_train_input2.npy  — shape (1200, 36)         — masque train
- Arkansas_train_labels.npy  — shape (1200,)              — labels train
- Arkansas_val_input1.npy    — shape (300, 10, 36)      — features validation
- Arkansas_val_input2.npy    — shape (300, 36)           — masque validation
- Arkansas_val_labels.npy    — shape (300,)                — labels validation
- Arkansas_test_input1.npy   — shape (8500, 10, 36)     — features test
- Arkansas_test_input2.npy   — shape (8500, 36)          — masque test
- Arkansas_test_labels.npy   — shape (8500,)               — labels test

- California_train_input1.npy — shape (1440, 10, 36)  
- California_train_input2.npy — shape (1440, 36)       
- California_train_labels.npy — shape (1440,)          
- California_val_input1.npy   — shape (360, 10, 36)   
- California_val_input2.npy   — shape (360, 36)        
- California_val_labels.npy   — shape (360,)            
- California_test_input1.npy  — shape (8200, 10, 36)  
- California_test_input2.npy  — shape (8200, 36)      
- California_test_labels.npy  — shape (8200,)           

---

## Specs d'interface — ce que ce module doit produire

Ces specs sont un contrat entre Point 4 et Point 5. Tout changement doit être discuté avec les deux parties.

### Tenseur X — Input 1
| Propriété | Valeur |
|-----------|--------|
| Forme | `(B, 10, 36)` — batch × bandes × timesteps |
| Type | `torch.float32` |
| Ordre des bandes | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 (index 0 à 9) |
| Normalisation | à définir ← *compléter ici* |

### Tenseur M — Input 2 (masque)
| Propriété | Valeur |
|-----------|--------|
| Forme | `(B, 36)` — batch × timesteps |
| Type | `torch.float32` |
| Valeurs | `1.0` = timestep valide, `0.0` = timestep manquant |
| Construction | `mask = (X.sum(dim=1) != 0).float()` — un timestep est manquant si **toutes** ses bandes valent 0 |

> **Note importante** : le masque doit être construit **avant** toute normalisation, car la normalisation peut modifier les valeurs 0.

### Labels Y
| Propriété | Valeur |
|-----------|--------|
| Forme | `(B,)` |
| Type | `torch.long` (requis par `nn.CrossEntropyLoss`) |
| Valeurs | entiers 0 à N_classes-1 (inchangés par rapport aux CSV) |

### Split train/val/test (Table 2 de l'article)
| Split | Arkansas | Californie |
|-------|----------|------------|
| Train | ~1 200 (240/classe × 5) | ~1 440 (240/classe × 6) |
| Val | ~300 (60/classe × 5) | ~360 (60/classe × 6) |
| Test | ~8 500 | ~8 200 |

Méthode : échantillonnage stratifié par classe, `random_state=42` pour la reproductibilité.

### Interface DataLoader attendue
```python
# Ce que Point 5 s'attend à recevoir à chaque batch :
for X, mask, y in dataloader:
    # X    : (B, 10, 36)  float32
    # mask : (B, 36)      float32
    # y    : (B,)         long
    ...
```

---

## Détails techniques — à compléter par Sarah

### Normalisation choisie
Division par 10 000 (X / 10000).
Les réflectances Sentinel-2 Level-2A sont stockées en entiers × 10 000.
Cette opération ramène les valeurs dans l'intervalle [0, 1].
Aucun Z-Score ni Min-Max appliqué — non mentionné dans le paper (Section 2.2.4).

### Gestion des données manquantes
Les 0 sont conservés sans imputation.
GEE marque un pas de temps manquant en mettant toutes ses bandes à 0 via unmask(0).
Le masque Input 2 est construit AVANT la normalisation sur les valeurs brutes :
un pas de temps est manquant si ses 10 bandes valent toutes 0.
Après normalisation, ces positions restent à 0.0.
Aucune interpolation appliquée — le module ALPE du modèle gère les manquants.
Taux de manquants : 23.83% pour Arkansas, 16.19% pour California.

### Autres transformations appliquées
vReshape : (N, 360) → (N, 36, 10) → transposé en (N, 10, 36)
pour respecter le format channels-first attendu par les Conv1D du modèle
(Figure 3 du paper : matrice 10×36, bandes × temps).
Split stratifié par classe avec random_state=42 (Table 2 du paper).
---

## Problèmes rencontrés & solutions
*à compléter par Coéquipier B*

---

## Décisions prises & pourquoi
*à compléter par Coéquipier B*

---

## Lien avec les autres points
- ← **Point 2** : consomme `Arkansas_10k.csv` et `California_10k.csv`
- ← **Point 3** : l'exploration a confirmé que les 0 sont des manquants (pas des vraies valeurs spectrales)
- → **Point 5** : fournit les tenseurs `(X, mask, y)` dans le format ci-dessus
