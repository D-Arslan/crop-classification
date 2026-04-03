# Documentation — ALPE + Transformer sub-module
## `transformer_alpe.py`

---

## Table des matières

1. [Contexte et rôle dans MCTNet](#1-contexte-et-rôle-dans-mctnet)
2. [Vue d'ensemble du fichier](#2-vue-densemble-du-fichier)
3. [sinusoidal_pe — encodage positionnel](#3-sinusoidal_pe--encodage-positionnel)
4. [ECA — Efficient Channel Attention](#4-eca--efficient-channel-attention)
5. [ALPE — Adaptive Learnable Positional Encoding](#5-alpe--adaptive-learnable-positional-encoding)
6. [TransformerSubModule](#6-transformersubmodule)
7. [Comment utiliser ce module](#7-comment-utiliser-ce-module)
8. [Ce qui reste à faire](#8-ce-qui-reste-à-faire)

---

## 1. Contexte et rôle dans MCTNet

### Le problème que ce module résout

MCTNet classe des pixels agricoles à partir de séries temporelles Sentinel-2. Chaque pixel est représenté par une matrice `10 bandes × 36 timesteps`. Le problème : certains timesteps sont **manquants** (à cause des nuages), représentés par des 0 dans les données.

Un encodage positionnel classique (comme dans le Transformer original) attribuerait quand même une position temporelle à ces timesteps vides, ce qui induirait le modèle en erreur. L'ALPE résout ce problème en **masquant les positions manquantes** avant de construire l'encodage positionnel.

### Place dans l'architecture globale

MCTNet est organisé en 3 **CTFusion stages** successifs. Chaque stage contient :
- Un **CNN sub-module** (fait par le coéquipier A) — extrait des patterns locaux
- Un **Transformer sub-module** (ce fichier) — capture les dépendances globales entre timesteps
- Un mécanisme de fusion des deux

```
Input : (B, 10, 36) + mask (B, 36)
         │
    ┌────┴────────────────────────┐
    │  CTFusion Stage 1           │
    │  CNN ──────┐                │
    │            ├── Fusion ──→ (B, 20, 18)
    │  Transformer+ALPE ──┘       │
    └─────────────────────────────┘
         │
    ┌────┴────────────────────────┐
    │  CTFusion Stage 2           │
    │  CNN ──────┐                │
    │            ├── Fusion ──→ (B, 40, 9)
    │  Transformer ──────┘        │
    └─────────────────────────────┘
         │
    ┌────┴────────────────────────┐
    │  CTFusion Stage 3           │
    │  CNN ──────┐                │
    │            ├── Fusion ──→ (B, 80, 1)
    │  Transformer ──────┘        │
    └─────────────────────────────┘
         │
    Global Max Pooling → (B, 80)
    MLP Classifier    → (B, N_classes)
```

**Règle importante** : l'ALPE n'est utilisé qu'au **Stage 1** (le seul où le masque est directement disponible en entrée du modèle).

---

## 2. Vue d'ensemble du fichier

Le fichier contient 4 éléments, du plus simple au plus complexe :

```
transformer_alpe.py
│
├── sinusoidal_pe()         ← fonction utilitaire (pas de paramètres entraînables)
├── class ECA               ← brique de base utilisée par ALPE
├── class ALPE              ← utilise sinusoidal_pe + ECA
└── class TransformerSubModule  ← utilise ALPE (optionnel) + MHA + FFN
```

### Convention de format

Dans tout ce fichier, les tenseurs suivent la convention `(B, C, T)` :
- `B` = taille du batch
- `C` = nombre de canaux (bandes spectrales, ou features après les premières couches)
- `T` = nombre de timesteps

Cette convention est différente du Transformer original qui utilise `(T, B, C)`. Le passage entre les deux se fait avec `.permute()`.

---

## 3. `sinusoidal_pe` — encodage positionnel

### Pourquoi un encodage positionnel ?

Le Transformer traite sa séquence d'entrée **sans notion d'ordre** — l'attention calcule des similarités entre tous les éléments sans savoir lequel est "avant" ou "après". Pour lui indiquer la position de chaque élément, on lui ajoute un signal qui dépend de la position.

### La formule sinusoïdale

Pour chaque position `t` (timestep) et chaque dimension `i` du vecteur de features :

```
PE(t, 2i)   = sin(t / 10000^(2i/C))
PE(t, 2i+1) = cos(t / 10000^(2i/C))
```

Les dimensions paires reçoivent un sinus, les impaires un cosinus. La fréquence diminue avec `i` : les premières dimensions oscillent vite (capturent des patterns courts), les dernières oscillent lentement (patterns longs).

### Ce que ça produit

Une matrice `(T, C)` — ici `(36, 10)` au Stage 1 — où chaque ligne est la "signature positionnelle" d'un timestep. Deux timesteps proches ont des signatures proches ; deux timesteps éloignés ont des signatures différentes.

### Implémentation

```python
def sinusoidal_pe(seq_len: int, d_model: int, device=None) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len).unsqueeze(1)          # (T, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2) * (-log(10000.0) / d_model)
    )                                                          # (C/2,)
    pe[:, 0::2] = torch.sin(position * div_term)              # colonnes paires
    pe[:, 1::2] = torch.cos(position * div_term[:d_model//2]) # colonnes impaires
    return pe  # (T, C)
```

`div_term` est le dénominateur `10000^(2i/C)` pré-calculé sous forme de décroissance exponentielle.

### Note sur l'utilisation

Cette fonction est appelée une seule fois dans `ALPE.__init__()`. Le résultat est stocké comme buffer fixe via `register_buffer` — il n'est donc pas recalculé à chaque `forward()`.

---

## 4. `ECA` — Efficient Channel Attention

### Rôle

ECA pondère les canaux d'un tenseur : certains canaux (bandes spectrales) sont jugés plus importants que d'autres pour le timestep courant, et leurs valeurs sont amplifiées en conséquence. C'est une forme d'attention, mais appliquée aux canaux plutôt qu'aux positions temporelles.

### Intuition

Imagine que tu regardes une série temporelle Sentinel-2. Pour détecter du riz en période de croissance, la bande NIR (B8) est très informative. Pour détecter des nuages résiduels, c'est plutôt le SWIR. ECA laisse le modèle apprendre automatiquement quel canal regarder en priorité selon le contexte.

### Architecture

```
x : (B, C, T)
    │
    ├── AdaptiveAvgPool1d(1) ──→ (B, C, 1)   ← résume chaque canal sur le temps
    │       squeeze(-1)       ──→ (B, C)
    │       unsqueeze(1)      ──→ (B, 1, C)   ← canal devient la séquence
    │
    ├── Conv1d(1→1, kernel=k) ──→ (B, 1, C)  ← capture les relations inter-canaux
    │   Sigmoid               ──→ (B, 1, C)   ← poids dans [0, 1]
    │       squeeze(1)        ──→ (B, C)
    │       unsqueeze(-1)     ──→ (B, C, 1)   ← prêt pour broadcast
    │
    └── x * poids             ──→ (B, C, T)   ← pondération
```

### Le kernel adaptatif

La taille du kernel Conv1D est calculée automatiquement à partir du nombre de canaux :

```python
t = int(abs(math.log2(channels) / gamma + b / gamma))
k = t if t % 2 else t + 1  # force impair pour avoir un centre
```

Avec `gamma=2, b=1` (valeurs de l'article ECA-Net) :
- C=10 → k=3
- C=20 → k=3
- C=40 → k=5

Un kernel impair est nécessaire pour que la convolution soit symétrique autour de la position centrale.

### Pourquoi Conv1D sur les canaux et non un simple FC ?

Une couche `Linear(C, C)` ferait l'attention en considérant **tous les canaux ensemble** (matrice C×C de paramètres). ECA utilise une Conv1D locale : chaque canal n'est mis en relation qu'avec ses voisins immédiats dans l'espace des canaux. Résultat : beaucoup moins de paramètres, et dans la pratique performances similaires ou meilleures.

---

## 5. `ALPE` — Adaptive Learnable Positional Encoding

### Le problème qu'il résout

Dans un Transformer standard, on ajoute le PE à toutes les positions, y compris les manquantes. Si le timestep `t=3` est absent (toutes les bandes à 0), lui attribuer quand même une position temporelle introduit du bruit : le modèle "voit" quelque chose là où il ne devrait rien voir.

L'ALPE dit : **avant d'encoder la position, efface les timesteps manquants du PE**. Ainsi le modèle ne reçoit un signal positionnel que pour les timesteps qui ont réellement des données.

### La formule de l'article

```
ALPE(t) = ECA( Conv1D( PE(t) ⊙ mask ) )
```

- `PE(t)` : encodage positionnel sinusoïdal, matrice `(T, C)`
- `⊙` : multiplication terme à terme
- `mask` : vecteur `(T,)` avec 1=valide, 0=manquant
- `Conv1D` : apprend des relations temporelles locales dans cet encodage masqué
- `ECA` : pondère les canaux de l'encodage résultant

Le résultat est **ajouté** aux features d'entrée x (connexion résiduelle).

### Déroulement du `forward()` pas à pas

Prenons un exemple concret : `B=2, C=10, T=36`, avec quelques timesteps manquants.

**Étape 1 — PE sinusoïdal**
```python
pe = sinusoidal_pe(T=36, C=10)  # → (36, 10)
```
Matrice fixe, identique pour chaque pixel du batch.

**Étape 2 — Mise en forme pour le broadcast**
```python
mask_bc = mask.unsqueeze(1)   # (B, T) → (B, 1, T)
pe_bct  = pe.T.unsqueeze(0)   # (36,10) → (10,36) → (1, 10, 36)
```
On prépare les deux tenseurs pour qu'ils soient compatibles en multiplication :
- `mask_bc` a un 1 sur la dimension C → se répète sur C canaux
- `pe_bct` a un 1 sur la dimension B → se répète sur le batch

**Étape 3 — Masquage**
```python
masked_pe = pe_bct * mask_bc  # (1, C, T) × (B, 1, T) → (B, C, T)
```
Pour chaque pixel du batch, les colonnes de PE correspondant aux timesteps manquants deviennent 0.

**Étape 4 — Conv1D**
```python
out = self.conv(masked_pe)  # (B, C, T) → (B, C, T)
out = self.bn(out)
out = F.relu(out)
```
Conv1D avec `kernel_size=3` et `padding=1` — la dimension T est conservée. Le modèle apprend à lisser et interpoler le PE masqué localement dans le temps.

**Étape 5 — ECA**
```python
out = self.eca(out)  # (B, C, T) → (B, C, T)
```
Pondère les canaux de l'encodage adaptatif.

**Étape 6 — Addition résiduelle**
```python
return x + out  # (B, C, T) + (B, C, T) → (B, C, T)
```
L'encodage positionnel adaptatif est ajouté aux features, sans les écraser.

### Paramètres entraînables d'ALPE
| Couche | Paramètres | Calcul |
|--------|-----------|--------|
| Conv1D (C=10, k=3) | 10×10×3 = 300 | poids uniquement (bias=False) |
| BatchNorm1d (C=10) | 2×10 = 20 | weight + bias |
| ECA Conv1d (k=3) | 1×1×3 = 3 | kernel adaptatif |
| **Total (Stage 1)** | **323** | |

---

## 6. `TransformerSubModule`

### Architecture complète

```
x : (B, C, T)  +  mask : (B, T)
│
├── [Si use_alpe=True] ALPE(x, mask) ──→ x  (B, C, T)
│
│   permute(0,2,1)  ──→  x_t : (B, T, C)
│
├── MultiHeadAttention(Q=x_t, K=x_t, V=x_t)
│   │
│   └── attn_out : (B, T, C)
│       x_t = LayerNorm(x_t + attn_out)    ← Add & Norm
│
├── FFN :
│   │   Linear(C → 4C)
│   │   ReLU
│   │   Dropout
│   │   Linear(4C → C)
│   │   Dropout
│   └── ffn_out : (B, T, C)
│       x_t = LayerNorm(x_t + ffn_out)     ← Add & Norm
│
│   permute(0,2,1)  ──→  out : (B, C, T)
│
└── return out
```

### Multi-Head Self-Attention (MHA)

L'attention permet à chaque timestep de "regarder" tous les autres timesteps et de pondérer leur importance. Avec n_head=5 têtes, le modèle fait cette opération en parallèle avec 5 sous-espaces différents, capturant plusieurs types de relations temporelles simultanément.

La Self-Attention signifie que Q, K et V sont tous les trois issus de la même séquence x_t :
```python
attn_out, _ = self.attn(x_t, x_t, x_t)  # Q=K=V=x_t
```

**Contrainte n_head** : `embed_dim` doit être divisible par `n_head`. Ici :
- Stage 1 : C=10, n_head=5 → 10/5=2 ✅
- Stage 2 : C=20, n_head=5 → 20/5=4 ✅
- Stage 3 : C=40, n_head=5 → 40/5=8 ✅

C'est pour ça que l'article choisit n_head=5 avec ces dimensions de canaux.

### Add & Norm

Après chaque sous-couche (attention ou FFN), on applique :
```python
x_t = LayerNorm(x_t + sous_couche_out)
```
- **Add** (connexion résiduelle) : évite le gradient vanishing dans les couches profondes
- **Norm** (LayerNorm) : normalise sur la dimension C pour chaque token indépendamment

### Feed Forward Network (FFN)

```
C → 4C → C
```

L'expansion à 4×C est une convention du Transformer original de Vaswani et al. L'article ne précise pas cette valeur, donc on suit la convention. Le FFN applique une transformation non-linéaire indépendamment à chaque timestep (pas d'interaction entre timesteps ici, c'est le rôle de la MHA).

### Paramètres entraînables par stage

| Composant | Stage 1 (C=10) | Stage 2 (C=20) | Stage 3 (C=40) |
|-----------|---------------|---------------|---------------|
| ALPE | 323 | — | — |
| MHA (Q,K,V projections + out) | 10×10×4 + 4×10 = 440 | 20×20×4 + 4×20 = 1 680 | 40×40×4 + 4×40 = 6 560 |
| LayerNorm ×2 | 2×(2×10) = 40 | 2×(2×20) = 80 | 2×(2×40) = 160 |
| FFN (Linear ×2) | 10×40+40 + 40×10+10 = 860 | 20×80+80 + 80×20+20 = 3 300 | 40×160+160 + 160×40+40 = 13 000 |
| **Total** | **~1 663** | **~5 060** | **~19 720** |
| **Total 3 stages** | | | **~26 443** |

(Les valeurs exactes peuvent varier légèrement selon le calcul des bias.)

---

## 7. Comment utiliser ce module

### Instanciation pour les 3 stages

```python
from transformer_alpe import TransformerSubModule

# Stage 1 — avec ALPE (masque disponible)
transformer_s1 = TransformerSubModule(
    channels=10,
    seq_len=36,
    n_head=5,
    use_alpe=True,
    kernel_size=3,
    dropout=0.1
)

# Stage 2 — sans ALPE
transformer_s2 = TransformerSubModule(
    channels=20,
    seq_len=18,
    n_head=5,
    use_alpe=False
)

# Stage 3 — sans ALPE
transformer_s3 = TransformerSubModule(
    channels=40,
    seq_len=9,
    n_head=5,
    use_alpe=False
)
```

### Appel dans le forward

```python
# Stage 1 : le masque est obligatoire
out_s1 = transformer_s1(x, mask)   # x:(B,10,36), mask:(B,36) → (B,10,36)

# Stages 2 et 3 : pas de masque
out_s2 = transformer_s2(x)         # x:(B,20,18) → (B,20,18)
out_s3 = transformer_s3(x)         # x:(B,40,9)  → (B,40,9)
```

### Contrat avec le CNN sub-module (coéquipier A)

Le Transformer reçoit la sortie du CNN et produit une sortie de **même forme**. Le CNN doit donc respecter :
- Entrée : `(B, C, T)` — même convention
- Sortie : `(B, C, T)` — même forme que l'entrée

La fusion des deux sorties (CNN + Transformer) sera implémentée dans le bloc CTFusion.

### Test de bon fonctionnement

```bash
python transformer_alpe.py
```

Doit afficher :
```
Stage 1 — Input : torch.Size([4, 10, 36])  Mask : torch.Size([4, 36])  Output : torch.Size([4, 10, 36])
Stage 2 — Input : torch.Size([4, 20, 18])  Output : torch.Size([4, 20, 18])
Stage 3 — Input : torch.Size([4, 40, 9])   Output : torch.Size([4, 40, 9])
Nombre total de paramètres (3 Transformers) : 26,433
Tous les tests sont passés.
```

---

## 8. Ce qui reste à faire

### 🔲 Prochaine étape : bloc CTFusion

Une fois le CNN du coéquipier A disponible, implémenter `CTFusion` qui :
1. Reçoit `(B, C, T)` + mask `(B, T)`
2. Passe dans CNN → `(B, C, T)`
3. Passe dans Transformer → `(B, C, T)`
4. Fusionne les deux → `(B, 2C, T//2)` (double les canaux, divise le temps par 2)

Le mécanisme exact de fusion (concaténation + Conv ? Addition ? Autre ?) n'est pas précisé dans l'article et devra être décidé lors de la prochaine session.
