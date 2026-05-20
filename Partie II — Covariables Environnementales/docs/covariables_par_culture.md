# Description des covariables environnementales par type de culture

Ce document décrit les 7 covariables retenues pour le projet (3 sol, 2 topographie, 2 climat),
avec pour chaque culture les valeurs typiques ou idéales observées dans la littérature.
Les cultures concernées sont celles des deux zones d'étude : **Arkansas** (soja, riz, maïs, coton)
et **Californie** (raisins, riz, luzerne, amandes, pistaches).

---

## 1. pH du sol

**C'est quoi ?**
Le pH mesure l'acidité ou l'alcalinité du sol. Il va de 0 (très acide) à 14 (très alcalin).
Un pH neutre est autour de 7. La plupart des cultures agricoles préfèrent un sol légèrement
acide à neutre (pH 5.5 à 7.5).

**Pourquoi c'est important pour classer les cultures ?**
Chaque culture pousse mieux dans un certain niveau de pH. Un sol trop acide ou trop
alcalin empêche les plantes d'absorber correctement les nutriments. Donc le pH du sol
aide à prédire quelles cultures sont présentes dans une zone.

| Culture | pH idéal | Valeurs typiques trouvées |
|---|---|---|
| **Soja** (Arkansas) | 6.0 – 7.0 | Sols neutres des plaines alluviales du Mississippi |
| **Riz** (Arkansas & Californie) | 5.5 – 7.0 | Tolère les sols légèrement acides et argileux |
| **Maïs** (Arkansas) | 5.8 – 7.0 | Sols riches, bien drainés, pH neutre |
| **Coton** (Arkansas) | 5.8 – 8.0 | Assez tolérant, pousse sur sols légèrement alcalins |
| **Raisins** (Californie) | 5.5 – 6.5 | Préfère les sols légèrement acides, bien drainés |
| **Luzerne** (Californie) | 6.5 – 7.5 | Exige un pH neutre à légèrement alcalin |
| **Amandes** (Californie) | 6.0 – 7.5 | Sols bien drainés, pH modéré |
| **Pistaches** (Californie) | 7.0 – 8.0 | Très tolérant au calcaire, préfère sols alcalins |

**Sources :**
- University of California Cooperative Extension — Crop Soil Requirements
- FAO, *Crop Production and Management*, 2021
- Ge, Y. et al. (2011). *Remote sensing of soil properties in precision agriculture: A review.* Frontiers of Earth Science.
- Dataset GEE : `OpenLandMap_SOL_SOL_PH-H2O_USDA-4C1A2A_M_v02`

---

## 2. Carbone organique du sol (OC)

**C'est quoi ?**
Le carbone organique (OC) mesure la quantité de matière organique dans le sol
(restes de plantes, animaux, microbes décomposés). Plus il y en a, plus le sol est fertile.
Il est mesuré en g/kg et multiplié par 5 dans le dataset GEE.

**Pourquoi c'est important pour classer les cultures ?**
Les cultures intensives comme le soja et le maïs se trouvent généralement sur des sols
très riches en matière organique (comme les alluvions d'Arkansas). Les cultures
arboricoles de Californie (amandes, pistaches) poussent souvent sur des sols pauvres.
Cela crée un contraste utile pour la classification.

| Culture | OC typique (g/kg) | Caractéristique du sol |
|---|---|---|
| **Soja** (Arkansas) | 15 – 30 | Sols très riches, plaines alluviales du Mississippi |
| **Riz** (Arkansas & Californie) | 10 – 25 | Sols humides, hydromorphes, bonne rétention |
| **Maïs** (Arkansas) | 15 – 30 | Sols fertiles, agriculture intensive |
| **Coton** (Arkansas) | 8 – 20 | Sols modérément riches |
| **Raisins** (Californie) | 5 – 15 | Sols assez pauvres, bien drainés |
| **Luzerne** (Californie) | 10 – 20 | Sols irrigués, modérément riches |
| **Amandes** (Californie) | 4 – 12 | Sols sableux à limoneux, pauvres |
| **Pistaches** (Californie) | 3 – 10 | Très peu exigeant, sols pauvres et secs |

**Sources :**
- Tufail et al. (2025). *Deep Learning Applications for Crop Mapping Using Multi-Temporal Sentinel-2 Data.* Remote Sensing — mentionne l'intérêt des données de sol auxiliaires.
- Jiao et al. (2025). *Multi-Layer Soil Moisture Profiling Based on BKA-CNN.* Agronomy — recommande d'intégrer les propriétés du sol dans les modèles.
- Dataset GEE : `OpenLandMap_SOL_SOL_ORGANIC-CARBON_USDA-6A1C_M_v02`

---

## 3. Texture du sol (classe USDA)

**C'est quoi ?**
La texture du sol décrit les proportions de sable, limon et argile. Le système USDA
classe les sols en 12 types (ex : argile, limon argileux, sable limoneux, etc.).
Plus un sol est argileux, plus il retient l'eau. Plus il est sableux, plus il draine vite.

**Pourquoi c'est important pour classer les cultures ?**
La texture détermine directement quelles cultures peuvent être cultivées. Le riz a besoin
d'un sol argileux pour retenir l'eau lors de l'inondation. Les amandes et pistaches ont
besoin de sols bien drainants pour éviter la pourriture des racines.

| Culture | Texture préférée | Raison principale |
|---|---|---|
| **Soja** (Arkansas) | Limon argileux / Argile limoneuse | Bonne rétention d'eau sans excès |
| **Riz** (Arkansas & Californie) | Argile lourde / Argile | Retient l'eau pour l'inondation des rizières |
| **Maïs** (Arkansas) | Limon / Limon sableux | Bon drainage mais rétention suffisante |
| **Coton** (Arkansas) | Limon sableux / Argile limoneuse | Tolère sols variés, préfère drainage modéré |
| **Raisins** (Californie) | Sable limoneux / Limon | Drainage rapide essentiel |
| **Luzerne** (Californie) | Limon / Argile limoneuse | Sol profond avec bonne capacité de rétention |
| **Amandes** (Californie) | Limon sableux / Sable | Sol léger, drainage excellent obligatoire |
| **Pistaches** (Californie) | Limon / Limon argileux | Tolère sols calcaires et peu fertiles |

**Sources :**
- Ou et al. (2024). *Improving soil organic carbon mapping in farmlands.* Environmental Sciences Europe — montre que la texture est un facteur clé de différenciation agricole.
- Demattê et al. (2018). *Soil organic carbon and texture retrieving and mapping using Sentinel-2.* Remote Sensing of Environment.
- Dataset GEE : `OpenLandMap_SOL_SOL_TEXTURE-CLASS_USDA-TT_M_v02`

---

## 4. Élévation (DEM)

**C'est quoi ?**
L'élévation mesure la hauteur du terrain par rapport au niveau de la mer, en mètres.
Elle est dérivée du modèle numérique de terrain (DEM SRTM à 30 m).

**Pourquoi c'est important pour classer les cultures ?**
Les cultures sont très liées à l'altitude. Le riz se cultive dans les basses plaines
inondables. Les vignes et les arbres fruitiers se trouvent souvent sur des coteaux
modérément élevés. Les zones de plaine plates sont dominées par les grandes cultures.

| Culture | Élévation typique (m) | Zone géographique |
|---|---|---|
| **Soja** (Arkansas) | 50 – 150 | Plaines alluviales basses, Mississippi |
| **Riz** (Arkansas) | 50 – 100 | Zones très basses, plaines inondables |
| **Maïs** (Arkansas) | 80 – 200 | Plaines agricoles légèrement vallonnées |
| **Coton** (Arkansas) | 80 – 150 | Plaines agricoles |
| **Riz** (Californie) | 0 – 50 | Vallée de Sacramento, très basse |
| **Raisins** (Californie) | 100 – 600 | Coteaux et vallées intérieures |
| **Luzerne** (Californie) | 50 – 300 | Vallées irriguées |
| **Amandes** (Californie) | 50 – 400 | Vallée centrale, plaines et coteaux |
| **Pistaches** (Californie) | 100 – 500 | Zones semi-arides, collines intérieures |

**Sources :**
- Adrah et al. (2025). *Integrating GEDI, Sentinel-2, and Sentinel-1 for tree crops mapping.* Remote Sensing of Environment — utilise le DEM SRTM pour corriger la topographie et améliorer la classification.
- Wang et al. (2024). MCTNet paper (papier du projet) — Arkansas et Californie présentent des structures agricoles très différentes liées au relief.
- Dataset GEE : SRTM Digital Elevation Model (NASA)

---

## 5. Landforms (formes du terrain)

**C'est quoi ?**
Les landforms classifient le type de relief : plaine, vallée, plateau, colline, pente, etc.
Ils sont dérivés du DEM via des algorithmes de géomorphologie (ex : Geomorpho90m).
Chaque pixel reçoit une étiquette de forme de terrain.

**Pourquoi c'est important pour classer les cultures ?**
Les cultures arboricoles (amandes, pistaches, vignes) sont souvent sur des collines
ou plateaux. Les grandes cultures (riz, soja, maïs) occupent les plaines et vallées.
La forme du terrain donne donc une information complémentaire à l'élévation seule.

| Culture | Landform typique | Description |
|---|---|---|
| **Soja / Maïs / Coton** (Arkansas) | Plaine alluviale | Terrain plat, très peu de relief |
| **Riz** (Arkansas & Californie) | Plaine inondable / Fond de vallée | Zones basses avec eau stagnante |
| **Raisins** (Californie) | Versant / Piémont | Pentes douces exposées au soleil |
| **Luzerne** (Californie) | Plaine irriguée / Terrasse | Zones plates avec accès à l'irrigation |
| **Amandes** (Californie) | Terrasse alluviale / Plaine | Sols plats mais bien drainés |
| **Pistaches** (Californie) | Colline / Plateau semi-aride | Zones élevées, peu d'eau disponible |

**Sources :**
- Adrah et al. (2025) — utilise la pente et la topographie pour filtrer les zones non-cultivées.
- Jiao et al. (2025) — recommande d'intégrer l'élévation et les formes de terrain dans les modèles de classification agricole.
- Dataset GEE : Geomorpho90m ou TAGEE (Terrain Analysis in Google Earth Engine)

---

## 6. Précipitation annuelle (BIO12 — WorldClim / CHELSA)

**C'est quoi ?**
La précipitation annuelle totale en mm/an. Elle indique combien d'eau tombe
en moyenne chaque année dans une zone. Elle est tirée des données climatiques
mondiales WorldClim ou CHELSA à 1 km de résolution.

**Pourquoi c'est important pour classer les cultures ?**
Les cultures irriguées de Californie (amandes, pistaches, luzerne) se trouvent dans des
zones très sèches (< 300 mm/an) car elles dépendent de l'irrigation artificielle.
Les cultures pluviales d'Arkansas (soja, maïs) se trouvent dans des zones plus humides
(> 1000 mm/an). Cette variable sépare bien les deux zones d'étude.

| Culture | Précipitation annuelle typique (mm) | Type d'agriculture |
|---|---|---|
| **Soja** (Arkansas) | 1100 – 1400 | Pluviale, pas d'irrigation nécessaire |
| **Riz** (Arkansas) | 1100 – 1400 | Irrigation complémentaire en été |
| **Maïs** (Arkansas) | 1000 – 1300 | Principalement pluvial |
| **Coton** (Arkansas) | 1000 – 1300 | Pluvial avec irrigation estivale |
| **Riz** (Californie) | 400 – 600 | Irrigation intensive requise |
| **Raisins** (Californie) | 250 – 600 | Zone méditerranéenne semi-aride |
| **Luzerne** (Californie) | 150 – 400 | Zones très sèches, 100% irrigué |
| **Amandes** (Californie) | 200 – 450 | Irrigation essentielle |
| **Pistaches** (Californie) | 150 – 350 | Culture la plus tolérante à la sécheresse |

**Sources :**
- Adrah et al. (2025) — utilise les zones agroclimatiques (basées sur la pluviométrie) pour expliquer la variation de performance entre régions.
- Ou et al. (2024) — montre que la précipitation annuelle (MAP) est le facteur climatique le plus important pour la distribution des cultures.
- Dataset GEE : WorldClim BIO12 ou CHIRPS

---

## 7. Température moyenne de la saison de croissance (BIO10)

**C'est quoi ?**
BIO10 est la température moyenne du trimestre le plus chaud (en °C). Elle représente
la chaleur disponible pendant la principale saison de croissance des cultures.

**Pourquoi c'est important pour classer les cultures ?**
Chaque culture a un besoin en chaleur différent. Le riz et le coton ont besoin de
températures élevées. Le blé et l'orge préfèrent les saisons fraîches. En Californie,
les pistaches et amandes nécessitent des hivers froids et des étés chauds (cycle de
dormance). Cette variable aide à séparer les cultures d'été des cultures d'hiver.

| Culture | Température saison de croissance (°C) | Besoin thermique |
|---|---|---|
| **Soja** (Arkansas) | 22 – 28 | Culture de saison chaude |
| **Riz** (Arkansas & Californie) | 25 – 32 | Culture tropicale, besoin élevé en chaleur |
| **Maïs** (Arkansas) | 22 – 30 | Culture d'été, besoin modéré à élevé |
| **Coton** (Arkansas) | 25 – 32 | Exige beaucoup de chaleur et de soleil |
| **Raisins** (Californie) | 20 – 28 | Climat méditerranéen, été chaud et sec |
| **Luzerne** (Californie) | 18 – 28 | Coupe multiple, tolère chaleur modérée |
| **Amandes** (Californie) | 22 – 30 | Besoin de chaleur pour la floraison |
| **Pistaches** (Californie) | 25 – 35 | Culture la mieux adaptée à la chaleur extrême |

**Sources :**
- Tufail et al. (2025) — montre l'importance du calendrier cultural et des conditions climatiques pour distinguer les cultures dans le temps.
- Wang et al. (2024). MCTNet paper — la phénologie différente entre cultures est liée aux conditions climatiques locales.
- Dataset GEE : WorldClim BIO10 ou CHELSA

---

## Résumé visuel — Quelles covariables séparent le mieux les cultures ?

| Covariable | Sépare Arkansas vs Californie | Sépare cultures dans Arkansas | Sépare cultures dans Californie |
|---|---|---|---|
| pH |  Moyen |  (riz acide vs coton alcalin) |  (pistaches alcalines vs raisins acides) |
| Carbone organique |  Fort |  (soja/maïs riches vs coton) |  (luzerne vs pistaches pauvres) |
| Texture |  Fort |  (riz argileux vs maïs limoneux) |  (riz argileux vs amandes sableux) |
| Élévation |  Fort |  Faible (tout est plat) |  (riz bas vs pistaches hauts) |
| Landforms |  Fort |  Faible |  (raisins/pistaches vs luzerne) |
| Précipitation |  Très fort |  Faible (zone homogène) |  (zones arides vs irriguées) |
| Température |  Moyen |  (riz/coton chauds vs maïs) |  (pistaches très chaudes vs raisins) |

---

## Références principales utilisées

1. **Wang et al. (2024)**. A lightweight CNN-Transformer network (MCTNet) for pixel-based crop mapping using time-series Sentinel-2. *Computers and Electronics in Agriculture*, 226, 109370.

2. **Tufail et al. (2025)**. Deep Learning Applications for Crop Mapping Using Multi-Temporal Sentinel-2 Data and Red-Edge Vegetation Indices. *Remote Sensing*, 17, 3207.

3. **Jiao et al. (2025)**. Multi-Layer Soil Moisture Profiling Based on BKA-CNN. *Agronomy*, 15, 2542.

4. **Adrah et al. (2025)**. Integrating GEDI, Sentinel-2, and Sentinel-1 for tree crops mapping. *Remote Sensing of Environment*, 319, 114644.

5. **Ou et al. (2024)**. Improving soil organic carbon mapping in farmlands using machine learning. *Environmental Sciences Europe*, 36.

6. **Ge, Y. et al. (2011)**. Remote sensing of soil properties in precision agriculture: A review. *Frontiers of Earth Science*, 5(3).

7. **OpenLandMap datasets** (pH, OC, Texture) — disponibles sur Google Earth Engine via `OpenLandMap_SOL_*`.

8. **WorldClim / CHELSA** — données climatiques BIO10 et BIO12 disponibles sur Google Earth Engine.
