# ADR-0001: Escopo, plano em fases e remoção do fork

- **Status:** Aceita (escopo, hardware e remoção confirmados pelo pesquisador em 2026-10-22)
- **Data:** 2026-10-22

## Contexto

O repositório continha um fork de [Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) (Apache-2.0), que classifica (a) a pose de uma única mão com um MLP sobre 21 landmarks e (b) a trajetória da ponta do indicador. Em 2024 foi feito sobre ele um protótipo do alfabeto datilológico (26 letras, 1.681 amostras, branch `feature/gui`).

Essa arquitetura não cobre a Libras:
- considera uma mão só, sem relação com o corpo (ponto de articulação) nem com o rosto (expressões não-manuais);
- trata o estático e o dinâmico como dois modelos separados, e o dinâmico usa só um ponto;
- usa a API legada do MediaPipe e TensorFlow (ver ADR-0003).

Parâmetros do projeto definidos pelo pesquisador:
- **Natureza:** pesquisa de laboratório universitário, com intenção de publicar artigo.
- **Vocabulário:** meta de longo prazo é o vocabulário completo da Libras; o início é um escopo reduzido.
- **Hardware de treino:** RTX 5090 local.

## Alternativas consideradas

1. **Evoluir o fork:** aproveita o código, mas a arquitetura (uma mão, dois modelos, TF/API legada) precisaria ser trocada quase inteira.
2. **Reescrever do zero e preservar o histórico em tags:** estrutura pensada para pesquisa (configs, experimentos reprodutíveis, documentação).

## Decisão

Reescrever do zero. O estado anterior fica preservado nas tags `v0-fork-original` (fe80f7b) e `v0-prototipo-alfabeto` (9e3957b), e a pasta `hand-gesture-recognition-using-mediapipe/` sai da árvore.

O trabalho segue estas fases:

| Fase | Entrega |
|---|---|
| 0. Fundação | Estrutura do repositório, `CLAUDE.md`, ADRs iniciais, ambiente |
| 1. Literatura e datasets | Fichas de datasets e fichamentos; escolha do(s) dataset(s) inicial(is) em ADR |
| 2. Extração | Vídeo → landmarks em cache, com ferramenta de inspeção visual de qualidade |
| 3. Baselines | MLP sobre estatísticas da sequência → GRU; avaliação independente de sinalizador |
| 4. Modelos e ablações | Transformer / grafos espaço-temporais (ST-GCN, SL-GCN); ablações de partes do corpo, normalização e aumento de dados |
| 5. Demo em tempo real | Webcam, janela deslizante, detecção de início e fim do sinal |
| 6. Extensões | Conjunto de teste próprio (requer aprovação do CEP), reconhecimento contínuo |

## Justificativa

O código antigo representa pouco esforço a preservar, e uma arquitetura inadequada contaminaria as decisões seguintes. As tags mantêm o histórico rastreável, e o protótipo do alfabeto pode ser citado como antecedente.

## Consequências

- **Escalar até o vocabulário completo** é uma questão de pesquisa em aberto. Com milhares de classes e poucas amostras por sinal, a classificação direta tende a falhar. Uma direção promissora, que pode virar contribuição do artigo, é **decompor o reconhecimento nos parâmetros fonológicos** (configuração de mão, ponto de articulação, movimento, orientação, expressões não-manuais), que são compartilhados entre sinais. Isso será investigado na Fase 1 e decidido em ADR próprio.
- A licença Apache-2.0 do código original só gera obrigações se algum trecho dele for reaproveitado.
