# Documentation — CTFusion block
## `src/ctfusion.py`

---

## Table des matières

1. [Contexte et rôle dans MCTNet](#1-contexte-et-rôle-dans-mctnet)
2. [Architecture détaillée](#2-architecture-détaillée)
3. [Décisions d'implémentation](#3-décisions-dimplémentation)
4. [Problèmes rencontrés](#4-problèmes-rencontrés)
5. [Comment utiliser ce module](#5-comment-utiliser-ce-module)
6. [Ce qui reste à faire](#6-ce-qui-reste-à-faire)

---

## 1. Contexte et rôle dans MCTNet

CTFusion est l'unité de base de MCTNet, répétée 3 fois successivement. Son rôle est double :

1. **Combiner les représentations CNN et Transformer** — le CNN capture les patterns locaux dans la série temporelle (ex: un pic de végétation sur quelques semaines), le Transformer capture les dépendances globales (ex: la relation entre le début et la fin de la saison). En les combinant, MCTNet bénéficie des deux types d'information.

2. **Réduire progressivement la dimension temporelle** — à chaque stage, le nombre de timesteps est divisé par 2 et le nombre de canaux est doublé. Le modèle "compresse" l'information temporelle au fil des stages.

### Progression des dimensions

| Stage | Entrée | Sortie | Canaux | Timesteps |
|-------|--------|--------|--------|-----------|
| 1 | (B, 10, 36) | (B, 20, 18) | 10 → 20 | 36 → 18 |
| 2 | (B, 20, 18) | (B, 40,  9) | 20 → 40 | 18 → 9 |
| 3 | (B, 40,  9) | (B, 80,  4) | 40 → 80 |  9 → 4 |

Après les 3 stages, le Global Max Pooling (dans `mctnet.py`) réduit `(B, 80, 4) → (B, 80)`, puis le MLP classifie.

---

## 2. Architecture détaillée

```
x : (B, C, T)  +  mask : (B, T)   [mask requis au Stage 1 uniquement]
│
├──────────────────────────────────────────┐
│  CNN sub-module                          │  Transformer sub-module
│  Conv1D → BN → Conv1D → BN → ReLU+res   │  [ALPE] → MHA → Add&Norm → FFN → Add&Norm
│  cnn_out : (B, C, T)                     │  tr_out : (B, C, T)
└──────────────────────┬───────────────────┘
                       │
          torch.cat([cnn_out, tr_out], dim=1)
                       │
                 fused : (B, 2C, T)
                       │
          MaxPool1d(kernel_size=2, stride=2)
                       │
                 out : (B, 2C, T//2)
```

**CNN et Transformer reçoivent la même entrée `x`** — ils travaillent en parallèle, pas en séquence.

---

## 3. Décisions d'implémentation

### Concaténation plutôt qu'addition
Les sorties CNN et Transformer sont concaténées sur la dimension des canaux (`dim=1`) plutôt qu'additionnées. L'article (Figure 3) montre explicitement une concaténation. L'addition aurait forcé les deux modules à produire des représentations dans le même espace — la concaténation leur laisse des espaces distincts, ce qui préserve mieux les informations complémentaires de chaque module.

### MaxPool1d pour la réduction temporelle
Conforme à la Figure 3 de l'article qui indique explicitement "Max Pooling" entre les stages. `MaxPool1d(kernel_size=2, stride=2)` sélectionne la valeur maximale sur chaque fenêtre de 2 timesteps — opération fixe, sans paramètres entraînables. Une Conv1D stride=2 aurait été une alternative apprenante (elle apprend la réduction au lieu de la faire mécaniquement), mais nous restons fidèles à l'article.

### CTFusion identique aux 3 stages
Le bloc CTFusion est strictement le même aux 3 stages — pas de cas spécial. La seule différence est `use_alpe=True` au Stage 1, géré par le paramètre du constructeur. Cette uniformité simplifie le code et est conforme à l'article.

### Global Max Pooling dans MCTNet, pas dans CTFusion
Le Stage 3 produit `(B, 80, 4)` — pas `(B, 80, 1)`. La réduction finale `(B, 80, 4) → (B, 80)` est réalisée par un Global Max Pooling dans `mctnet.py`. Garder cette responsabilité hors de CTFusion respecte le principe de séparation des responsabilités : CTFusion fait la fusion + réduction intermédiaire, MCTNet fait l'agrégation finale.

---

## 4. Problèmes rencontrés

### T=9 au Stage 3 ne donne pas T=1
**Constat** : `MaxPool1d(2)` sur T=9 donne `floor(9/2) = 4`, pas 1.

**Mauvaise solution écartée** : mettre un `AdaptiveAvgPool1d(1)` dans CTFusion Stage 3 pour forcer T=1. Cela aurait mélangé la responsabilité de CTFusion (fusion locale) avec celle du Global Max Pooling (agrégation finale).

**Solution retenue** : CTFusion Stage 3 sort `(B, 80, 4)` — normal et attendu. Le Global Max Pooling dans `mctnet.py` réduit ensuite `(B, 80, 4) → (B, 80)` en une seule opération. C'est cohérent avec la Figure 3 de l'article qui place le Global Max Pooling après les 3 stages CTFusion.

---

## 5. Comment utiliser ce module

### Instanciation

```python
from src.ctfusion import CTFusion

stage1 = CTFusion(in_channels=10, seq_len=36, n_head=5, use_alpe=True,  kernel_size=3)
stage2 = CTFusion(in_channels=20, seq_len=18, n_head=5, use_alpe=False, kernel_size=3)
stage3 = CTFusion(in_channels=40, seq_len=9,  n_head=5, use_alpe=False, kernel_size=3)
```

### Forward

```python
# Stage 1 : masque obligatoire
out1 = stage1(x, mask)   # (B, 10, 36) → (B, 20, 18)

# Stages 2 et 3 : pas de masque
out2 = stage2(out1)      # (B, 20, 18) → (B, 40,  9)
out3 = stage3(out2)      # (B, 40,  9) → (B, 80,  4)

# Global Max Pooling (dans mctnet.py)
# out3.max(dim=2).values  →  (B, 80)
```

### Lancer les tests

```bash
# depuis le dossier "Point 5 — Model Implementation"
python -m tests.test_ctfusion
```

---

## 6. Ce qui reste à faire

- 🔲 Implémenter `src/mctnet.py` — assemble les 3 CTFusion + Global Max Pooling + MLP classifier
- 🔲 Implémenter `train.py` — boucle d'entraînement + métriques OA/Kappa/F1
- 🔲 Entraîner sur Arkansas et Californie
- 🔲 Évaluer et comparer avec la Table 5 de l'article
