# Point 4 — Data Preprocessing

## Statut : ⏳ En cours

## Objectif
Transformer les CSV bruts (Arkansas_10k.csv, California_10k.csv) en tenseurs PyTorch prêts à être consommés par MCTNet. Produire un DataLoader utilisable directement par le Point 5.

## Fichiers produits
- *à compléter par Sarah*

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
*à compléter*

### Gestion des données manquantes
*à compléter — noter ici si une imputation est appliquée ou si les 0 sont conservés*

### Autres transformations appliquées
*à compléter*

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
