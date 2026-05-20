# Avancement — Deep Learning for Crop Classification
## M1 SII — USTHB — 2025/2026

> **Statut final : ✅ Projet livré.** Ce document est figé et conservé comme historique de travail. La structure actuelle du repo est décrite dans le `README.md` principal.

**Article** : "A lightweight CNN-Transformer network for pixel-based crop mapping using time-series Sentinel-2 imagery" — Wang et al., 2024

---

## Équipe

| Membre | Rôle |
|--------|------|
| Arslan | Point 5 — ALPE + Transformer sub-module |
| Tesnime | Point 5 — CNN sub-module |
| Sarah | Point 4 — Data Preprocessing |

---

## État d'avancement

| Point | Intitulé | Responsable | Statut |
|-------|----------|-------------|--------|
| 1 | Literature Review | Équipe | ✅ Terminé |
| 2 | Dataset Acquisition | Équipe | ✅ Terminé |
| 3 | Data Exploration | Équipe | ✅ Terminé |
| 4 | Data Preprocessing | Sarah | ✅ Terminé |
| 5 | Model Implementation | Arslan + Tesnime | ✅ Terminé |
| 6 | Partie 2 — Covariables environnementales | Tesnime + Sarah | ✅ Terminé |
| 7 | Partie 3 — Contributions (GatedMCTNet, Multiscale, MCTNetUSkip/UNet) | Arslan + Tesnime | ✅ Terminé |

---

## Organisation Git

### Branches
| Branche | Rôle |
|---------|------|
| `main` | Code stable — version remise au prof uniquement |
| `dev` | Intégration — on merge ici après validation |
| `arslan/transformer` | ALPE + Transformer + CTFusion (Arslan) |
| `tesnime/cnn` | CNN sub-module (Tesnime) |
| `sarah/preprocessing` | Data Preprocessing (Sarah) |

### Règles
- Ne jamais pusher directement sur `main` ou `dev`
- Chacun travaille sur sa branche, ouvre une PR vers `dev` quand c'est prêt
- `dev` → `main` uniquement aux jalons clés (fin Point 5, avant soutenance)

### Convention des commits
```
[module] description courte

Exemples :
[transformer] fix PE buffer dans __init__
[cnn] add residual connection
[ctfusion] implement stride 2 fusion
[data] add mask construction
[docs] update point5.md
```

### Workflow quotidien
```bash
# Début de session
git checkout arslan/transformer
git pull origin dev

# Fin de session
git add src/fichier.py
git commit -m "[transformer] description"
git push origin arslan/transformer
```

---

## Prochaines étapes

- [x] Sarah — preprocessing terminé, deux scales livrées (scale30 + scale20) ✅
- [x] Sarah — dataset Arkansas corrigé et équilibré (10 000 pixels, 2000/classe) ✅
- [x] Arslan — CTFusion implémenté et testé ✅
- [x] Arslan — MCTNet assemblé et testé (56 798 params, cible article 55 059) ✅
- [x] Arslan — train.py implémenté et testé ✅
- [x] Arslan — train.py mis à jour pour scale30/scale20 ✅
- [ ] Tesnime — confirmer que le CNN est final, ouvrir PR tesnime/cnn -> dev
- [ ] Tesnime — lancer l'entraînement sur sa machine (voir message ci-dessous)
- [ ] Lancer entraînement complet Arkansas + Californie × 2 scales (200 époques chacun)
- [ ] Comparer OA/Kappa/F1 avec Table 5 de l'article

---

## Détails

Le détail complet de chaque point (données, décisions techniques, divergences avec l'article) est dans :
**`Partie I/resume.md`**
