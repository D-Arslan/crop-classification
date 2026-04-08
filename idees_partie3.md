# Idées pour la Partie 3 — Améliorations et extensions

Ce fichier recense les pistes d'amélioration identifiées pendant l'implémentation de la Partie 1.
Chaque idée est accompagnée de sa motivation et de ce qu'elle apporterait au projet.

---

## 1. MLP à deux couches vs une couche (Section 2.3.2)

**Contexte** : l'article décrit un MLP à une seule couche `Linear(80 → N_classes)`.
Nous avons implémenté cette version fidèle. Une alternative à deux couches
`Linear(80→80) → ReLU → Dropout → Linear(80→N_classes)` est plus standard en pratique.

**Expérience proposée** : entraîner les deux versions sur Arkansas et Californie,
comparer OA / Kappa / F1. Si la version à deux couches améliore les résultats,
cela constitue une contribution originale par rapport à l'article.

**Fichiers concernés** : `src/mctnet.py` — paramètre `mlp_hidden` optionnel.

---

## 2. Dropout dans le Transformer

**Contexte** : l'article ne mentionne pas explicitement de dropout dans le Transformer.
Nous avons mis `dropout=0.1` par convention. Tester `dropout=0.0` vs `0.1` vs `0.2`
pourrait améliorer les résultats, surtout sur Californie (plus difficile, OA=0.852).

**Expérience proposée** : grid search sur `dropout` ∈ {0.0, 0.05, 0.1, 0.2}.

---

## 3. Remplacement du MaxPool par une Conv1D stride=2

**Contexte** : nous avons utilisé `MaxPool1d(2)` conforme à l'article (Figure 3).
Une `Conv1D(stride=2)` à la place apprendrait la réduction temporelle au lieu de
la faire mécaniquement — potentiellement plus expressive.

**Expérience proposée** : remplacer `MaxPool1d` par `Conv1D(2C→2C, stride=2)`
dans CTFusion, comparer les performances et le nombre de paramètres.

---

## 4. Gestion des classes déséquilibrées

**Contexte** : Arkansas a 45% de Soybeans et seulement 1.7% de Others.
Californie a 36.5% de Rice. Ce déséquilibre peut biaiser le modèle
vers les classes majoritaires.

**Expérience proposée** : utiliser `CrossEntropyLoss(weight=class_weights)`
avec des poids inversement proportionnels à la fréquence de chaque classe.
Mesurer l'impact sur le F1 macro (métrique la plus sensible au déséquilibre).

---

## 5. Data augmentation temporelle

**Contexte** : l'article ne mentionne pas de data augmentation.
Des augmentations adaptées aux séries temporelles pourraient améliorer
la généralisation, surtout avec peu d'échantillons d'entraînement (240/classe).

**Idées** :
- Décalage temporel aléatoire (jitter ±1 timestep)
- Masquage aléatoire de timesteps supplémentaires (simule plus de nuages)
- Mélange de deux pixels de la même classe (Mixup)

---

## 6. Comparaison avec d'autres modèles de la littérature

**Contexte** : l'article compare MCTNet à plusieurs baselines (LSTM, Transformer pur,
CNN pur, etc.). Reproduire une ou deux de ces baselines permettrait de valider
notre pipeline d'évaluation et de contextualiser nos résultats.

**Modèles suggérés** :
- LSTM simple (baseline temporelle classique)
- Transformer pur sans CNN (ablation de la branche CNN)
