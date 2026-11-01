import numpy as np

def sigmoide(s):
    return 1 / (1 + np.exp(-np.clip(s, -500, 500)))

def derivada_sigmoide(y):
    return y * (1 - y)

def treinar_mlp(X, Y, neuronios=32, eta=0.1, paciencia=0.01, lim_epocas=10000):
    historico_epocas, historico_erros = [], []
    epoca = 0
    erro_total = 100

    num_amostras = X.shape[0]
    tam_entrada = X.shape[1]
    tam_saida = Y.shape[1]

    np.random.seed(42)
    w1 = np.random.uniform(-0.5, 0.5, (tam_entrada, neuronios))
    w2 = np.random.uniform(-0.5, 0.5, (neuronios, tam_saida))
    b1 = np.random.uniform(-0.5, 0.5, (1, neuronios))
    b2 = np.random.uniform(-0.5, 0.5, (1, tam_saida))

    print("Iniciando treinamento...")

    while erro_total > paciencia and epoca < lim_epocas:
        erro_total = 0

        for i in range(num_amostras):
            s1 = np.dot(X[i, :], w1) + b1
            func_atv = sigmoide(s1)
            s2 = np.dot(func_atv, w2) + b2
            y_pred = sigmoide(s2)

            y_true = Y[i, :].reshape(1, tam_saida)

            erro = 0.5 * np.sum((y_true - y_pred)**2)
            erro_total += erro

            delta_saida = (y_true - y_pred) * derivada_sigmoide(y_pred)
            delta_camada_oculta = np.dot(delta_saida, w2.T) * derivada_sigmoide(func_atv)

            w1 += eta * np.dot(X[i, :].reshape(-1, 1), delta_camada_oculta)
            w2 += eta * np.dot(func_atv.T, delta_saida)
            b1 += eta * delta_camada_oculta
            b2 += eta * delta_saida
        
        epoca += 1
        historico_epocas.append(epoca)
        historico_erros.append(erro_total)

        if epoca % 100 == 0:
            print(f"Época: {epoca} - Erro Total: {erro_total:.4f}")
        
    return w1, w2, b1, b2, historico_epocas, historico_erros

def prever_mlp(X, w1, w2, b1, b2):
    s1 = np.dot(X, w1) + b1
    func_atv = sigmoide(s1)
    s2 = np.dot(func_atv, w2) + b2
    return sigmoide(s2)


def treinar_naive_bayes(X, Y):

    if Y.ndim > 1:
        Y = np.argmax(Y, axis=1)

    classes = np.unique(Y) #pega as classes que tem
    num_amostras, num_features = X.shape #num_amostras = quantidade de linhas, num_features = quantidade de colunas
    num_classes = len(classes)

    prioris = np.zeros(num_classes)
    medias = np.zeros((num_classes, num_features))
    desvios = np.zeros((num_classes, num_features))

    print("Iniciando treinamento...")

    for i, c in enumerate(classes):
        X_c = X[Y == c]
        medias[i, :] = np.mean(X_c, axis=0)
        desvios[i, :] = np.std(X_c, axis=0) + 0.15
        prioris[i] = len(X_c) / num_amostras

    return medias, desvios, prioris, classes    

def prever_naive_bayes(X, medias, desvios, prioris, classes):
    previsoes = []

    for x in X:
        pontuacoes = []

        for i in range(len(classes)):
            media = medias[i, :]
            desvio = desvios[i, :]

            parte_log = np.log(1 / (desvio * np.sqrt(2 * np.pi)))
            parte_exp = - 0.5 * ((x - media) / desvio) ** 2
            likelihood = parte_log + parte_exp
            pontuacao = np.log(prioris[i] + 1e-9) + np.sum(likelihood)
            pontuacoes.append(pontuacao)
        
        previsoes.append(classes[np.argmax(pontuacoes)])

    return np.array(previsoes)

