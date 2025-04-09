

---

# Sentiment Analysis for Movies using NLP

## Description

Ce projet vise à analyser les sentiments des critiques de films à l'aide de techniques de traitement du langage naturel (NLP). Nous avons utilisé plusieurs modèles d'apprentissage automatique pour prédire si une critique de film est positive ou négative. 

### Modèles utilisés :
- **Régression Logistique**
- **LSTM (Long Short-Term Memory)**
- **SVC (Support Vector Classifier)**

## Objectifs

1. **Prétraitement des données** : Nettoyer et transformer les critiques de films brutes en un format adapté pour l'entraînement des modèles.
2. **Entraînement des modèles** : Utilisation de trois approches différentes (Régression Logistique, LSTM et SVC) pour entraîner des modèles de classification des sentiments.
3. **Évaluation des performances** : Comparer les performances des modèles sur un jeu de données de test en utilisant des métriques telles que la précision, le rappel et le F1-score.
4. **Optimisation des modèles** : Ajuster les hyperparamètres et expérimenter avec différents prétraitements pour améliorer la précision du modèle.

## Prérequis

Assurez-vous d'avoir les bibliothèques suivantes installées :

- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow` (pour LSTM)
- `matplotlib`
- `seaborn`
- `nltk` (pour le prétraitement du texte)
- `joblib` (pour sauvegarder et charger les modèles)

Vous pouvez installer ces dépendances en utilisant `pip` :

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn nltk joblib
```

## Structure du projet

Le projet est structuré comme suit :

```
Sentiment_Analysis_Movies/
├── data/
│   ├── train_data.csv      # Jeu de données d'entraînement
│   ├── test_data.csv       # Jeu de données de test
├── models/
│   ├── logistic_regression_model.pkl  # Modèle de régression logistique entraîné
│   ├── lstm_model.h5                 # Modèle LSTM entraîné
│   ├── svc_model.pkl                  # Modèle SVC entraîné
├── notebooks/
│   ├── sentiment_analysis.ipynb       # Notebook pour l'analyse exploratoire des données                  # Script pour entraîner le modèle SVC
├── README.md                          # Ce fichier
└── requirements.txt                    # Liste des dépendances
```

## Méthodologie

### Prétraitement des données

- Tokenisation des critiques.
- Suppression des stopwords et des caractères spéciaux.
- Transformation du texte en vecteurs numériques (par exemple, avec TF-IDF ou embeddings).

### Entraînement des modèles

Les modèles sont entraînés à l'aide des scripts suivants :
- `train_logistic_regression.py` : Entraîne le modèle de régression logistique.
- `train_lstm.py` : Entraîne un modèle LSTM pour capturer les dépendances temporelles dans les critiques de films.
- `train_svc.py` : Entraîne un modèle SVC, qui est une méthode classique de classification.

### Évaluation des modèles

Une fois les modèles entraînés, vous pouvez les évaluer sur le jeu de données de test à l'aide des métriques suivantes :
- Précision
- Rappel
- F1-score

### Sauvegarde et chargement des modèles

Les modèles entraînés sont sauvegardés dans le dossier `models/` :
- Le modèle de régression logistique est enregistré sous forme de fichier `.pkl`.
- Le modèle LSTM est enregistré au format `.h5`.
- Le modèle SVC est également sauvegardé en format `.pkl`.



## Résultats et Comparaison des Modèles
