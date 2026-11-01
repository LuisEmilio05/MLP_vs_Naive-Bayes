import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def exibir_metricas_por_classe(matriz_confusao, nomes_classes):
    num_classes = len(nomes_classes)
    total_amostras = np.sum(matriz_confusao)
    total_acertos = 0

    print("\n" + "-" * 70)
    print(f"{'Classe':<15} | {'Precisão':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Qtd. Amostras':<8}")
    print("\n" + "-" * 70)
    
    for k in range(num_classes):
        tp = matriz_confusao[k, k]
        fp = np.sum(matriz_confusao[:, k]) - tp
        fn = np.sum(matriz_confusao[k, :]) - tp
        
        quantidade_amostras = tp + fn
        total_acertos += tp
        precisao = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0.0

        nome = nomes_classes[k]
        print(f"{nome:<15} | {precisao:<10.2f} | {recall:<10.2f} | {f1:<10.2f} | {quantidade_amostras:<8}")        

    acuracia = total_acertos / total_amostras if total_amostras > 0 else 0.0
    print("-" * 70)
    print(f"Acurácia Global: {acuracia:.4f}")
    print("-" * 70 + "\n")


def salvar_matriz_confusao(matriz_confusao, nomes_classes, nome):
    nome_arquivo = "matriz_" + nome + ".png"
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao, display_labels=nomes_classes)
    disp.plot(cmap="Blues", values_format="d", ax=ax, xticks_rotation=90)
    plt.title("Matriz de Confusão - (Conjunto de Teste)")
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=300)
    print(f"Figura da matriz guardada em: '{nome_arquivo}'")
    plt.show()


def salvar_curva_erro(epocas, perdas, nome):
    nome_arquivo = "perda_" + nome + ".png"
    plt.figure(figsize=(8, 5))
    plt.plot(epocas, perdas, color="darkblue", linewidth=1.5)
    plt.title("Curva de Aprendizagem - MLP")
    plt.xlabel("Épocas")
    plt.ylabel("Erro Total (Loss)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=300)
    print(f"Figura da curva de loss guardada em: '{nome_arquivo}'")
    plt.show()
