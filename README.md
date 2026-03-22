# Laboratorio Tecnico 05: Treinamento Fim-a-Fim do Transformer

Este repositorio contem a entrega do Laboratorio 05 do iCEV. O objetivo e instanciar o modelo Transformer criado anteriormente, conecta-lo a um dataset real do Hugging Face e escrever o Loop de Treinamento (Forward, Loss, Backward, Step). O foco nao e criar um tradutor de producao, mas provar que a arquitetura consegue aprender, forcando a funcao de perda (Loss) a cair significativamente ao longo das epocas.

## Bibliotecas Utilizadas

* Python 3.x
* torch (PyTorch)
* datasets (Hugging Face)
* transformers (Hugging Face)

## Como rodar o codigo

1. Clone este repositorio no seu ambiente local.
2. E recomendado usar o seu ambiente virtual ja configurado:
```bash
source venv/bin/activate
```

3. Garanta que as bibliotecas necessarias estejam instaladas:
```bash
pip install torch datasets transformers
```

4. A execucao esta dividida nos dois scripts principais do motor de otimizacao e teste:
```bash
# Para rodar o Loop de Treinamento (Tarefa 3)
python3 tarefa3.py

# Para rodar a Prova de Fogo de memorizacao (Tarefa 4)
python3 tarefa4.py
```

## O que o script faz

Diferente do laboratorio 04 que focou na construcao matematica "from scratch" usando NumPy, este projeto integra o modelo a um pipeline de Deep Learning real usando Diferenciacao Automatica (Autograd) do PyTorch. O sistema executa quatro tarefas principais:

1. Preparando o Dataset Real (Tarefa 1): Integra a biblioteca `datasets` para baixar o `multi30k` (Ingles-Alemao). Filtra um subconjunto minusculo de 1.000 frases para garantir que o treinamento rode de forma rapida na CPU.

2. Tokenizacao Basica (Tarefa 2): Utiliza a biblioteca `transformers` para importar o tokenizador pre-treinado `bert-base-multilingual-cased`. Converte os textos em matrizes de IDs, aplicando padding para uniformizar o comprimento das tensores e inserindo tokens especiais de inicio e fim.

3. Motor de Otimizacao (Tarefa 3): Roda o Training Loop acoplado a classe Transformer (traduzida para herdar de `torch.nn.Module`). Passa os lotes pelo Encoder/Decoder usando Teacher Forcing, calcula o erro com `CrossEntropyLoss` (ignorando os tokens de padding artificial) e aplica `loss.backward()` e `optimizer.step()` via otimizador Adam para atualizar os pesos.

4. A Prova de Fogo - Overfitting Test (Tarefa 4): Isola uma unica frase do conjunto de treino e forca a rede a memoriza-la decorando o padrao matricial atraves de 40 epocas de treino intenso. Apos isso, chama a funcao de inferencia auto-regressiva para "vomitar" a traducao exata daquela frase token por token, provando o fluxo correto dos gradientes.

## Logica Matematica (Requisito do Contrato Pedagogico)

(Preencha esta secao com as suas palavras explicando como a reducao do erro e feita descendo a superficie de gradientes, ou como o otimizador Adam atualiza as matrizes Wq, Wk e Wv durante o backpropagation, conforme exigido nas instrucoes).

## Validacao de Resultados

Ao rodar os scripts, o console exibira:

* O shape dos tensores padronizados gerados pelo tokenizador (ex: tensores de tamanho 32).
* O log iterativo das epocas de treinamento, mostrando o valor da Loss caindo drasticamente (indicando a convergencia do modelo).
* No teste de overfitting, a exibicao da frase original em Ingles, o gabarito em Alemao e a sequencia gerada iterativamente pelo modelo demonstrando a memorizacao bem sucedida.

## Versionamento

Conforme as regras de entrega, este commit avaliado possui a tag ou release v1.0.

## Nota de Credito

Partes geradas/complementadas com IA, revisadas por Anderson.

Conforme as regras da disciplina e o contrato pedagogico sobre o uso de IA Generativa neste laboratorio especifico, registro que utilizei o assistente virtual para:
* Facilitar a manipulacao da biblioteca `datasets` e a aplicacao da tokenizacao em lote nas Tarefas 1 e 2.
* Auxiliar na transposicao da arquitetura Transformer construida no Lab 04 (que estava em NumPy puro) para classes baseadas em `torch.nn.Module`, viabilizando o calculo de gradientes (`loss.backward()`) exigido na Tarefa 3.
* Estruturar a organizacao e modularizacao dos arquivos do projeto.

