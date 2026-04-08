# Avancement — Deep Learning for Crop Classification
## M1 SII — USTHB — 2025/2026

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
| 4 | Data Preprocessing | Sarah | ⏳ En cours |
| 5 | Model Implementation | Arslan + Tesnime | ⏳ En cours |

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

- [ ] Sarah — finaliser le preprocessing, compléter `Partie I/Point 4 — Data Preprocessing/point4.md`
- [ ] Tesnime — terminer le CNN sub-module
- [x] Arslan — CTFusion implémenté et testé ✅
- [ ] Tesnime — confirmer que le CNN est final, ouvrir PR tesnime/cnn -> dev
- [ ] Arslan — implémenter mctnet.py (assemblage complet)
- [ ] Arslan — implémenter train.py (boucle entraînement + métriques)
- [ ] Ensemble — assembler MCTNet complet, entraîner, évaluer

---

## Détails

Le détail complet de chaque point (données, décisions techniques, divergences avec l'article) est dans :
**`Partie I/resume.md`**
