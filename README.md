# Projet_ML
Projet machine learning intitulé : Prédiction Simultanée de la Réhospitalisation et du Changement de Traitement chez les Patients Diabétiques par Classification Multi-Label 
## Description
Ce projet consiste à construire un pipeline complet de machine learning appliqué à un dataset médical de patients diabétiques. L’objectif est de prédire deux variables cibles :
- La réadmission des patients
- Le changement de traitement

## Structure du projet
- Nettoyage des données
- Feature Engineering
- Prétraitement
- Modélisation
- Évaluation

## Dataset
Le dataset contient des informations médicales sur les patients, incluant des données démographiques, des diagnostics, des traitements et des visites hospitalières.

## Préparation des données
- Gestion des valeurs manquantes
- Suppression des variables non pertinentes ou redondantes
- Transformation des variables catégorielles
- Création de nouvelles variables pertinentes (feature engineering)

## Approche de modélisation
- Classification multi-label avec ClassifierChain
- Modèles utilisés :
  - Random Forest
  - XGBoost
  - LightGBM

## Métriques d’évaluation
- ROC-AUC
- F1-score (Macro)
- Precision et Recall

## Résultats
Les modèles ont été comparés à l’aide du ROC-AUC moyen sur les deux cibles. Le modèle XGBoost a obtenu les meilleures performances dans ce projet.

## Perspectives
- Optimisation des hyperparamètres avec GridSearchCV et RandomizedSearchCV
- Amélioration de la gestion du déséquilibre des classes
- Exploration de nouveaux modèles
- Amélioration du feature engineering
- Déploiement du modèle

## Prérequis
- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- imbalanced-learn

## Utilisation
1. Exécuter le script de nettoyage des données
2. Exécuter le script de feature engineering
3. Exécuter le script de modélisation
4. Analyser les résultats
