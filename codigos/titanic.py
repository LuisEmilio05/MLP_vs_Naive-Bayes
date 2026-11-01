import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from modelos import treinar_mlp, prever_mlp, treinar_naive_bayes, prever_naive_bayes
from metricas import salvar_matriz_confusao, salvar_curva_erro, exibir_metricas_por_classe

if __name__ == "__main__":
    df = pd.read_csv('titanic.csv')

    codificacao_sobrevive = pd.get_dummies(df['Survived'], dtype=float)
    Y = codificacao_sobrevive.values
    nomes_classes = ['Não Sobreviveu', 'Sobreviveu']

    colunas_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    codificacao_sexo = pd.get_dummies(df['Sex'], dtype=float)
    X_temp = pd.concat ([df[colunas_features], codificacao_sexo], axis=1)
    X = X_temp.values
    #X = df[colunas_features].values

    X = np.nan_to_num(X, nan=0.0)
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X = (X - X_min) / np.where((X_max - X_min) == 0, 1.0, (X_max - X_min))

    y_labels = df['Survived'].values

    # divisao dos dados (80% - treino, 20% - teste)
    X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.20, random_state=42, stratify=y_labels)
    

    w1, w2, b1, b2, epocas, perdas = treinar_mlp(X_treino, Y_treino, 32, 0.1, 0.01, 10000) # neuronios, eta, paciencia, limite_epocas

    y_pred_prob = prever_mlp(X_teste, w1, w2, b1, b2)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_true_classes = np.argmax(Y_teste, axis=1)

    matriz_confusao = confusion_matrix(y_true_classes, y_pred_classes, labels=range(len(nomes_classes)))
    exibir_metricas_por_classe(matriz_confusao, nomes_classes)

    salvar_curva_erro(epocas, perdas, "Titanic_MLP")
    salvar_matriz_confusao(matriz_confusao, nomes_classes, "Titanic_MLP")
    
    medias, desvios, prioris, classes = treinar_naive_bayes(X_treino, Y_treino)
    y_pred_nb = prever_naive_bayes(X_teste, medias, desvios, prioris, classes)
    y_true_nb = np.argmax(Y_teste, axis=1)

    matriz_confusao = confusion_matrix(y_true_nb, y_pred_nb, labels=range(len(nomes_classes)))
    exibir_metricas_por_classe(matriz_confusao, nomes_classes)
    salvar_matriz_confusao(matriz_confusao, nomes_classes, "Titanic_NB")
    