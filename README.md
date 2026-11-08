# Projeto de Classificação: MLP e Gaussian Naive Bayes

Este repositório contém a implementação dos modelos de aprendizado de máquina Multi-Layer Perceptron (MLP) e Gaussian Naive Bayes desenvolvidos em Python, avaliados nos conjuntos de dados Titanic e Pokémon.

---

## Integrantes do Grupo
Trabalho acadêmico desenvolvido em colaboração pelos integrantes:

*   **Integrante 1** - Luís Emílio Gonçalves Mendes
*   **Integrante 2** - Nicholas Fernandez Garcia

---

## 1. Estrutura de Arquivos

Para a correta execução dos experimentos, garanta que os arquivos de código e as bases de dados estejam no mesmo diretório:

```text
.
├── modelo.py                 # Implementação dos modelos (MLP, Naive Bayes)
├── metricas.py               # Funções de matriz de confusão e métricas
├── titanic.py                # Script de execução do dataset Titanic
├── pokemon.py                # Script de execução do dataset Pokémon
├── titanic.csv               # Base de dados do Titanic
└── pokemon.csv               # Base de dados dos Pokémon
```

---

## 2. Requisitos e Instalação

O projeto requer **Python 3.8** ou superior. As dependências necessárias podem ser instaladas via terminal com o comando:

```bash
pip install numpy pandas matplotlib
```

---

## 3. Carregamento das Bases de Dados

As bases são carregadas automaticamente pelos scripts através do Pandas, buscando os arquivos `.csv` no diretório de execução:

* **Titanic (`titanic.csv`):** Carregado em `titanic.py`. O script realiza o pré-processamento das variáveis contínuas e categóricas (como sexo e classe), preenche valores faltantes e converte variáveis categóricas de texto (como Sexo) em valores numéricos e aplica One-Hot Encoding no _target_ (Y) da rede neural.
* **Pokémon (`pokemon.csv`):** Carregado em `pokemon.py`. O script seleciona os 9 atributos de combate numéricos (HP, Attack, Defense, etc.), normaliza os dados com Min-Max e utiliza divisão estratificada (*stratified split*) sobre o atributo `Type_1` (18 classes).

Certifique-se de que os nomes dos arquivos `.csv` correspondam aos indicados acima antes de iniciar a execução.

---

## 4. Como Executar os Experimentos

### Experimento 1: Titanic (Classificação Binária)
Execute o script correspondente no terminal:

```bash
python titanic.py
```

* **Saídas geradas:**
  * Métricas de Precisão, Recall, F1-Score e Acurácia Global impressas no terminal para ambos os modelos.
  * Curva de convergência do erro: `perda_Titanic_MLP.png`.
  * Matriz de confusão da MLP: `matriz_Titanic_MLP.png`.
  * Matriz de confusão do NB: `matriz_Titanic_NB.png`.

---

### Experimento 2: Pokémon (Classificação Multiclasse - 18 Classes)
Execute o script correspondente no terminal:

```bash
python pokemon.py
```

* **Saídas geradas:**
  * Tabela completa de métricas por classe para as 18 categorias de Pokémon.
  * Curva de erro por época: `perda_Pokemon_MLP.png`.
  * Matriz de confusão $18 \times 18$: `matriz_Pokemon_MLP.png`.
  * Matriz de confusão do NB: `matriz_Pokemon_NB.png`.

## Artigo
Se deseja saber mais sobre o nosso projeto, acesse o documento abaixo:

*   [📘 Artigo Científico - (PDF)](documentos/Relatorio_Tecnico.pdf)
























