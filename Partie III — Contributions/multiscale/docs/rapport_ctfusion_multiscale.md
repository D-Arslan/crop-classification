# CTFusion MultiScale — Documentation technique
## Fichier : `ctfusion_MultiScale.py`
### Projet : Deep Learning for Crop Classification — Partie 3 (Modèle amélioré)

---

## 1. Rôle du fichier

Ce fichier définit le bloc **CTFusion** modifié utilisé dans l'architecture améliorée de la Partie 3. Par rapport à la version originale de Wang et al. (2024), **une seule modification** est apportée : la branche CNN est remplacée par un **MSCNNSubModule** (Multi-Scale CNN), tandis que la branche Transformer et la structure globale du bloc restent identiques.

```
Branche originale (Part 1) :   CNNSubModule      (2 conv. 1D, kernel fixe)
Branche améliorée (Part 3) :   MSCNNSubModule    (convolutions multi-échelles)
```

---

## 2. Structure du bloc CTFusion

### 2.1 Flux de données

```
Entrée x : (B, C, T)
            │
       ┌────┴────┐
       │         │
       ▼         ▼
  MSCNNSubModule   TransformerSubModule
  (B, C, T)        (B, C, T)
       │         │
       └────┬────┘
            │  concat(dim=1)
            ▼
        (B, 2C, T)
            │
       MaxPool1d(k=2, s=2)
            ▼
        (B, 2C, T//2)
```

Les deux branches (**CNN** et **Transformer**) traitent **la même entrée `x` en parallèle**. Leurs sorties, de même shape `(B, C, T)`, sont concaténées sur la dimension des canaux pour produire `(B, 2C, T)`, puis le MaxPool1d divise la dimension temporelle par 2.

### 2.2 Dimensions par stage dans MCTNet

| Stage | Entrée | Sortie |
|-------|--------|--------|
| Stage 1 | (B, 10, 36) | (B, 20, 18) |
| Stage 2 | (B, 20, 18) | (B, 40, 9) |
| Stage 3 | (B, 40, 9) | (B, 80, 4) |

Le Global Max Pooling final `(B, 80, 4) → (B, 80)` est appliqué dans `mctnet.py`, pas dans ce fichier.

---

## 3. Composants du bloc

### 3.1 MSCNNSubModule — branche CNN multi-échelle (nouveauté Part 3)

Importé depuis `src.cnn_submodule`, ce module remplace le `CNNSubModule` original. Il capture des **dépendances temporelles à différentes échelles** simultanément en utilisant des convolutions de kernels variés (par exemple k=3, k=5, k=7) en parallèle, dont les sorties sont fusionnées.

**Justification scientifique :** dans une série temporelle agricole de 36 pas de 10 jours, certains événements phénologiques sont courts (floraison, ~1–2 semaines → kernel petit), d'autres s'étendent sur plusieurs semaines (montaison, maturation → kernel grand). Une convolution à kernel fixe ne peut capturer qu'une seule échelle temporelle. L'approche multi-échelle est inspirée des architectures InceptionTime et TSception, couramment utilisées pour les séries temporelles.

**Interface** : même que `CNNSubModule` — prend `(B, C, T)`, retourne `(B, C, T)`.

### 3.2 TransformerSubModule — branche attention temporelle (inchangée)

Importé depuis `src.transformer_alpe`. Identique à la version de Wang et al. (2024) :
- **ALPE** (Adaptive Local Positional Encoding) activé uniquement au Stage 1 via `use_alpe=True`
- Attention multi-tête (`n_head=5` par défaut)
- FFN avec dimension cachée `8 × channels`
- LayerNorm sur chaque sous-couche

Le paramètre `mask (B, T)` est requis au Stage 1 pour que l'ALPE masque les pas de temps nuageux.

### 3.3 MaxPool1d — réduction temporelle

```python
self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
```

Conforme à la Figure 3 de Wang et al. (2024). Divise `T` par 2 à chaque stage, ce qui donne la progression 36 → 18 → 9 → 4 sur les dimensions temporelles.

---

## 4. Interface de la classe `CTFusion`

### Paramètres du constructeur

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `in_channels` | int | — | Nombre de canaux en entrée (10, 20 ou 40) |
| `seq_len` | int | — | Longueur de la séquence temporelle (36, 18 ou 9) |
| `n_head` | int | 5 | Nombre de têtes d'attention multi-tête |
| `kernel_size` | int | 3 | Kernel du CNN et de l'ALPE |
| `use_alpe` | bool | False | Activer l'ALPE (Stage 1 seulement) |
| `dropout` | float | 0.1 | Taux de dropout dans le Transformer |

### Méthode `forward`

```python
def forward(x: Tensor, mask: Tensor | None = None) -> Tensor
```

| Argument | Shape | Requis |
|----------|-------|--------|
| `x` | `(B, C, T)` | Toujours |
| `mask` | `(B, T)` | Uniquement si `use_alpe=True` (Stage 1) |
| **Retour** | `(B, 2C, T//2)` | — |

---

## 5. Différence par rapport à la version originale (Part 1)

| Aspect | CTFusion original (Part 1) | CTFusion MultiScale (Part 3) |
|--------|---------------------------|------------------------------|
| Branche CNN | `CNNSubModule` (2 conv. résiduel, kernel fixe k=3) | `MSCNNSubModule` (multi-échelle, plusieurs kernels) |
| Branche Transformer | `TransformerSubModule` + ALPE | Identique |
| MaxPool | `MaxPool1d(k=2, s=2)` | Identique |
| Structure globale | Parallèle → Concat → Pool | Identique |
| Dimensions d'entrée/sortie | `(B, C, T) → (B, 2C, T//2)` | Identique |

La modification est **minimale et ciblée** : seul le sous-module CNN change. Cela permet une comparaison directe et équitable avec le modèle baseline lors de l'évaluation.

---

## 6. Utilisation dans MCTNet (Part 3)

Dans `mctnet.py`, les trois stages sont instanciés comme suit :

```python
self.stage1 = CTFusion(10, 36, n_head, kernel_size, use_alpe=True,  dropout=dropout)
self.stage2 = CTFusion(20, 18, n_head, kernel_size, use_alpe=False, dropout=dropout)
self.stage3 = CTFusion(40,  9, n_head, kernel_size, use_alpe=False, dropout=dropout)
```

L'`MSCNNSubModule` reçoit automatiquement le bon `in_channels` (10, 20, 40) à chaque stage, et doit maintenir la dimension de sortie en canaux identique à l'entrée pour que la concaténation produise bien `2C` canaux.

---

## 7. Points d'attention pour le rapport

**Justification du choix architectural :** le remplacement du CNN simple par un CNN multi-échelle est motivé par la **nature multi-fréquence de la phénologie agricole**. Les cultures présentent des événements temporels à plusieurs échelles : début de saison (transitions rapides), phase de croissance (tendance longue), sénescence (déclin progressif). Un receptive field variable permet au modèle de les capturer simultanément.

**Comparabilité avec la baseline :** puisque seul le module CNN change, toute différence de performance entre Part 1 et Part 3 est directement attribuable à l'apport du multi-scale. C'est une bonne pratique d'ablation scientifique.

**Complexité paramétrique :** l'ajout de kernels supplémentaires augmente légèrement le nombre de paramètres. Il convient de rapporter le nombre total de paramètres du modèle amélioré et de le comparer à celui du modèle baseline (~XXk paramètres, à extraire via `sum(p.numel() for p in model.parameters())`).

---

## Références

- Wang et al. (2024). *A lightweight CNN-Transformer network for pixel-based crop mapping using time-series Sentinel-2 imagery.* Computers and Electronics in Agriculture, 226, 109370. — Architecture originale CTFusion, Figure 3.
- Ismail Fawaz et al. (2020). *InceptionTime: Finding AlexNet for time series classification.* Data Mining and Knowledge Discovery. — Inspiration pour les convolutions multi-échelles sur séries temporelles.
