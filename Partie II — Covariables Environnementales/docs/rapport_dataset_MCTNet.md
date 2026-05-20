# Rapport de projet — Compréhension du Dataset
## Réseau MCTNet pour la cartographie des cultures par séries temporelles Sentinel-2

---

> **Contexte :** Ce rapport présente une analyse détaillée du jeu de données utilisé dans l'article *"A lightweight CNN-Transformer network for pixel-based crop mapping using time-series Sentinel-2 imagery"* (Wang et al., 2024). L'objectif est de comprendre la nature des données d'entrée, leur structure et leur signification physique.

---

## 1. Vue d'ensemble du dataset

Le dataset est construit à partir de **trois sources de données complémentaires** :

| Source | Rôle | Résolution | Fournisseur |
|--------|------|------------|-------------|
| Sentinel-2 Level-2A | Images satellitaires (données d'entrée du modèle) | 10–20 m | ESA |
| Cropland Data Layer (CDL) | Vérité terrain — étiquettes des types de cultures | 30 m | USDA |
| ESA WorldCover 2021 | Masque des zones non agricoles | 10 m | ESA |

Les zones d'étude sont deux états américains aux caractéristiques agricoles très différentes : l'**Arkansas** (grandes exploitations, cultures homogènes) et la **Californie** (petites parcelles, grande diversité de cultures).

---

## 2. Comprendre les bandes spectrales

### 2.1 Qu'est-ce qu'une bande spectrale ?

La lumière solaire est réfléchie différemment par chaque type de surface. L'œil humain ne perçoit que trois couleurs (rouge, vert, bleu). Un satellite comme Sentinel-2, en revanche, est capable de mesurer la lumière réfléchie sur **13 plages de longueurs d'onde distinctes**, appelées **bandes spectrales**, dont certaines sont totalement invisibles à l'œil nu.

Chaque bande capture donc une information physique particulière sur la surface observée.

```
Lumière solaire réfléchie par la végétation
                ↓
┌──────────────────────────────────────────────────────┐
│  Bleu │ Vert │ Rouge │ Red Edge │  NIR  │    SWIR    │
│  B2   │  B3  │  B4   │ B5,6,7,8A│  B8  │  B11, B12  │
└──────────────────────────────────────────────────────┘
  ←── visible par l'œil humain ───→  ←── invisible ──→
```

### 2.2 Les 10 bandes retenues et leur utilité

Les auteurs ont exclu les bandes 1, 9 et 10 (résolution de 60 m, trop grossière). Les **10 bandes retenues** sont :

| Bande | Nom commun | Résolution | Ce qu'elle détecte |
|-------|------------|------------|---------------------|
| B2 | Bleu | 10 m | Couleur de la plante, atmosphère |
| B3 | Vert | 10 m | Reflet de la chlorophylle |
| B4 | Rouge | 10 m | Absorption par la chlorophylle |
| B5 | Red Edge 1 | 20 m | Transition végétation/sol, stress |
| B6 | Red Edge 2 | 20 m | Teneur en chlorophylle |
| B7 | Red Edge 3 | 20 m | Structure du couvert végétal |
| B8 | NIR | 10 m | Densité de la végétation, biomasse |
| B8A | Red Edge 4 | 20 m | Santé et vigueur de la plante |
| B11 | SWIR 1 | 20 m | Teneur en eau de la végétation |
| B12 | SWIR 2 | 20 m | Stress hydrique, humidité du sol |

### 2.3 Pourquoi ces bandes sont-elles utiles pour distinguer les cultures ?

Le principe fondamental est que chaque type de culture **réfléchit la lumière différemment** selon son stade de croissance :

- Une **plante saine et dense** réfléchit fortement dans le NIR et absorbe le rouge → NDVI élevé.
- Une **plante stressée** réfléchit moins dans le NIR → NDVI faible.
- Les **bandes Red Edge** sont particulièrement sensibles aux variations fines de l'état de santé des plantes, ce qui les rend très utiles pour distinguer des espèces proches.

```
Exemple : distinguer maïs et soja en Arkansas

            Maïs          Soja
NIR :       fort en juillet   fort en août-septembre
Red Edge :  pic précoce       pic tardif
SWIR :      diminue vite      reste stable plus longtemps
```

En combinant les 10 bandes, le modèle dispose d'une **signature spectrale unique** pour chaque type de culture.

---

## 3. La dimension temporelle — les séries temporelles

### 3.1 Pourquoi utiliser des séries temporelles ?

Une seule image satellitaire à un instant donné ne suffit pas pour distinguer les cultures. Par exemple, en été, le maïs et le soja peuvent avoir des réflectances similaires. C'est en observant **l'évolution dans le temps** que les différences deviennent claires :

- Le maïs atteint son pic de végétation **plus tôt** dans l'année que le soja.
- Le riz nécessite une inondation préalable des parcelles, visible dans les premières semaines.
- Les arbres fruitiers (amandes, pistaches) ont des cycles phénologiques très différents des cultures annuelles.

### 3.2 Construction des séries temporelles : étape par étape

**Étape 1 — Collecte brute**

Toutes les images Sentinel-2 disponibles sur l'année 2021 sont téléchargées depuis Google Earth Engine. Avec un passage tous les 5 jours, cela représente théoriquement ~73 images, mais beaucoup sont inutilisables à cause des nuages.

**Étape 2 — Élimination des pixels nuageux**

Chaque pixel affecté par un nuage est détecté et supprimé grâce au masque de nuages fourni avec les images Sentinel-2.

**Étape 3 — Agrégation par fenêtres de 10 jours**

L'année est découpée en **36 fenêtres de 10 jours**. Pour chaque fenêtre, la **valeur médiane** des observations valides (sans nuages) est calculée.

```
Exemple pour une fenêtre de 10 jours (Band 8 - NIR) :

  Passage 1 → ☁️ nuage     → éliminé
  Passage 2 → ✅ val = 0.62 → conservé
  Passage 3 → ✅ val = 0.58 → conservé  
  Passage 4 → ✅ val = 0.60 → conservé
  Passage 5 → ☁️ nuage     → éliminé

  Valeurs valides : [0.62, 0.58, 0.60]
  → Médiane = 0.60  ✅ (valeur représentative de la fenêtre)
```

**Pourquoi la médiane et pas la moyenne ?**

La médiane est **plus robuste** face aux valeurs aberrantes. Si un pixel nuageux passe à travers le filtre (nuage non détecté), sa valeur sera anormalement basse et fausserait une moyenne. La médiane, elle, n'est pas affectée par ces valeurs extrêmes :

```
Valeurs : [0.62, 0.58, 0.60, 0.05 ← nuage non détecté]

Moyenne  = (0.62 + 0.58 + 0.60 + 0.05) / 4 = 0.46  ← faussée ❌
Médiane  = 0.59                                      ← fiable  ✅
```

**Étape 4 — Gestion des fenêtres vides (données manquantes)**

Certaines fenêtres de 10 jours peuvent ne contenir **aucune** observation valide (exemple : région très nuageuse en hiver). Dans ce cas, la valeur est marquée **0** et constitue ce que les auteurs appellent une **"donnée manquante"**. C'est précisément pour gérer ces cas que le module ALPE a été conçu.

### 3.3 Structure finale d'un échantillon

Après ce traitement, chaque pixel de la zone d'étude est représenté par une matrice :

```
          Temps (36 pas de temps, 1 tous les 10 jours)
          Jan        Fév  ...  Juin  ...  Nov        Déc
        ┌────────────────────────────────────────────────┐
   B2   │  0.05  0.06  0    0.05  ...  0.03  0.04  0.05 │
   B3   │  0.08  0.09  0    0.08  ...  0.05  0.06  0.07 │
   B4   │  0.06  0.07  0    0.06  ...  0.04  0.04  0.05 │
   B5   │  0.12  0.13  0    0.14  ...  0.08  0.09  0.10 │
   B6   │  0.20  0.22  0.21 0.25  ...  0.15  0.16  0.18 │
   B7   │  0.25  0.28  0.26 0.32  ...  0.18  0.19  0.21 │
   B8   │  0.35  0.40  0    0.55  ...  0.28  0.30  0.32 │
   B8A  │  0.30  0.35  0    0.48  ...  0.24  0.26  0.28 │
   B11  │  0.18  0.20  0.19 0.22  ...  0.14  0.15  0.16 │
   B12  │  0.10  0.12  0.11 0.13  ...  0.08  0.09  0.10 │
        └────────────────────────────────────────────────┘
               0 = données manquantes (nuages)

Taille de la matrice : 10 bandes × 36 temps = 360 valeurs
```

---

## 4. Constitution des échantillons étiquetés

### 4.1 Processus d'échantillonnage

10 000 points ont été tirés **aléatoirement** dans chaque état, en respectant deux filtres :
- **Filtre CDL** : confiance ≥ 95 % pour garantir la qualité de l'étiquette.
- **Masque WorldCover** : uniquement les zones classifiées comme terres cultivées.

### 4.2 Répartition des classes

Les classes représentant moins de 5 % des échantillons ont été regroupées en "Autres".

**Arkansas :**

| Classe | Échantillons totaux | Train | Validation | Test |
|--------|-------------------|-------|------------|------|
| Soja | 4 677 | 240 | 60 | 4 377 |
| Riz | 2 423 | 240 | 60 | 2 123 |
| Maïs | 1 522 | 240 | 60 | 1 222 |
| Coton | 762 | 240 | 60 | 462 |
| Autres | 616 | 240 | 60 | 316 |
| **Total** | **10 000** | **1 200** | **300** | **8 500** |

**Californie :**

| Classe | Échantillons totaux | Train | Validation | Test |
|--------|-------------------|-------|------------|------|
| Raisins | 2 054 | 240 | 60 | 1 754 |
| Riz | 2 037 | 240 | 60 | 1 737 |
| Luzerne | 974 | 240 | 60 | 674 |
| Amandes | 783 | 240 | 60 | 483 |
| Pistaches | 640 | 240 | 60 | 340 |
| Autres | 3 512 | 240 | 60 | 3 212 |
| **Total** | **10 000** | **1 440** | **360** | **8 200** |

### 4.3 Stratégie de partitionnement

Un point important est que **300 échantillons par classe** ont été sélectionnés pour former les ensembles d'entraînement et de validation (ratio 80/20), indépendamment du nombre total d'échantillons disponibles. Le reste — souvent très majoritaire — constitue l'ensemble de test. Cela permet d'évaluer les modèles sur des données représentatives de la vraie distribution géographique.

---

## 5. Résumé visuel du pipeline de construction du dataset

```
Données brutes Sentinel-2 (toute l'année 2021)
         ↓
   Suppression des pixels nuageux
         ↓
   Calcul de la médiane par fenêtre de 10 jours
         ↓
   36 pas de temps × 10 bandes = matrice 360 valeurs / pixel
         ↓
   Masquage WorldCover (ne garder que les zones agricoles)
         ↓
   Échantillonnage aléatoire de 10 000 points / état
         ↓
   Étiquetage via CDL (confiance ≥ 95%)
         ↓
   Partitionnement Train / Validation / Test
         ↓
   Dataset final prêt pour l'entraînement de MCTNet
```

---

## 6. Conclusion

Le dataset utilisé dans cet article est construit de manière rigoureuse pour refléter les conditions réelles de la télédétection agricole. L'utilisation de **10 bandes spectrales complémentaires** permet de capturer des informations physiologiques variées sur les cultures. La construction de **36 pas de temps** via agrégation médiane garantit une représentation temporelle régulière et robuste aux nuages. Enfin, la présence inévitable de **données manquantes** (valeurs = 0) constitue un défi central que l'architecture MCTNet, et plus particulièrement son module ALPE, tente de résoudre sans recourir à des techniques d'interpolation coûteuses.

---

*Référence : Wang et al. (2024). A lightweight CNN-Transformer network for pixel-based crop mapping using time-series Sentinel-2 imagery. Computers and Electronics in Agriculture, 226, 109370.*
