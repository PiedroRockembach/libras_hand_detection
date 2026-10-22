# 2. Trabalhos relacionados

*Primeira versão escrita a partir da revisão da Fase 1 (2026-10-22). Detalhes, números e protocolos estão nos fichamentos em [docs/literatura/](../literatura/README-indice.md) e nas fichas em [docs/datasets/](../datasets/README.md). Números só são comparáveis quando o dataset e o protocolo (dependente ou independente de sinalizador) coincidem.*

## 2.1 Datasets

Os datasets públicos de Libras para sinais isolados têm **vocabulário pequeno e poucos sinalizadores**:
- MINDS-Libras: 20 sinais, 12 sinalizadores, CC BY 4.0;
- LIBRAS-UFOP: 56 sinais organizados em pares mínimos, 5 sinalizadores, acesso mediante acordo.

O V-LIBRASIL tem vocabulário amplo (1.364 termos), mas só 3 articuladores, com um exemplo de cada sinal por articulador, e problemas de formato documentados (vídeos em câmera lenta, legenda gravada na imagem). O MALTA-LIBRAS (PUCRS) reúne vídeos de fontes variadas e foi usado como teste fora de domínio. Não há dataset público de qualidade do alfabeto datilológico que inclua as letras com movimento.

Em outras línguas existem conjuntos grandes com protocolo independente de sinalizador:
- ASL Citizen: 2.731 sinais, 52 sinalizadores;
- AUTSL: 226 sinais, 43 sinalizadores;
- PopSign: 250 sinais, 47 sinalizadores.

O WLASL, muito citado, **não** tem divisão independente de sinalizador.

## 2.2 Reconhecimento baseado em esqueleto e landmarks

- **Modelos de grafo.** Os grafos espaço-temporais (ST-GCN, SL-GCN) são o estado da arte para entrada de esqueleto. O SL-GCN supera LSTM e transformer com a mesma entrada MediaPipe na comparação do OpenHands ([Selvaraj 2022](../literatura/selvaraj-2022-openhands.md); [Jiang 2021](../literatura/jiang-2021-sam-slr.md)).
- **Modelos leves com MediaPipe.** Em landmarks MediaPipe, modelos leves de convolução 1D + transformer venceram a competição Kaggle ISLR (2023) ([Sohn 2023](../literatura/sohn-2023-kaggle-islr.md)). A receita deles combina:
  - subconjunto de pontos;
  - normalização robusta a pontos ausentes;
  - velocidade e aceleração como *features*;
  - aumentos com *flip* + troca de mãos, reamostragem e mascaramento temporal.
- **Peso do estimador de pose.** Só trocar o estimador para o MediaPipe elevou o SPOTER de 63,18% para 78,29% no WLASL100 ([Bohacek 2022](../literatura/bohacek-2022-spoter-mediapipe.md)).

## 2.3 Reconhecimento de Libras

- **Protocolo de referência.** O protocolo mais rigoroso em uso é o *nested leave-one-person-out* (LOPO):
  - [Alves 2024](../literatura/alves-2024-skeleton-image.md), com OpenPose e imagem de esqueleto: 0,93 no MINDS-Libras e 0,82 no LIBRAS-UFOP;
  - [dos Santos 2025](../literatura/santos-2025-landmark-subset.md), com MediaPipe: 0,94 ± 0,04 e 0,91 ± 0,04.
- **Dois achados de dos Santos 2025 que pesam no nosso desenho:**
  - usar os 543 pontos do MediaPipe degrada muito o resultado, então é preciso escolher um subconjunto de pontos;
  - imputar pontos ausentes por *spline* dá ganhos grandes.
- **RGB.** Um I3D chega a 92,5% independente de sinalizador no MINDS ([Costa 2025](../literatura/costa-2025-i3d-libras.md)). Distilar um modelo RGB para um modelo de landmarks reduz a diferença entre os dois ([Costa 2026](../literatura/costa-2026-distilacao-mamba.md)).
- **Generalização.** Modelos treinados no MINDS generalizam mal para o MALTA-LIBRAS ([Delucis 2025](../literatura/delucis-2025-malta-libras.md)). Ou seja, acurácia alta no MINDS não implica robustez.
- **O que não existe.** Não foram encontrados trabalhos de Libras com vocabulário grande, supervisão fonológica ou pré-treino em pose.

## 2.4 Escalar o vocabulário

- **Supervisão fonológica auxiliar.** Ensinar o modelo a prever também os parâmetros do sinal (configuração de mão, local, movimento) melhora o reconhecimento em vocabulário grande: +8,7 pontos com SL-GCN no WLASL2000 ([Kezar 2023](../literatura/kezar-2023-phonology.md)). O ganho é maior com poucos exemplos por sinal ([Sem-Lex](../literatura/kezar-2023-sem-lex.md)). Propriedades fonológicas podem ser reconhecidas até em sinais não vistos no treino ([WLASL-LEX](../literatura/tavella-2022-wlasl-lex.md)).
- **Pré-treino auto-supervisionado em pose** ajuda no regime de poucos exemplos ([SignBERT+](../literatura/hu-2023-signbert-plus.md), [OpenHands](../literatura/selvaraj-2022-openhands.md), [Uni-Sign](../literatura/li-2025-uni-sign.md)).
- **Busca *one-shot* em dicionário.** Com um encoder de pose pré-treinado em ASL, foi possível buscar sinais num dicionário de 10.235 sinais com um vídeo cada ([Vandendriessche 2025](../literatura/vandendriessche-2025-one-shot.md)). É um caminho viável para "todo o vocabulário" usando dicionários de Libras.

## 2.5 RGB versus pose

- **Com estimadores antigos**, o RGB vence ([WLASL](../literatura/li-2020-wlasl.md); [ASL Citizen](../literatura/desai-2023-asl-citizen.md): I3D 63,1% contra ST-GCN/MediaPipe 59,5%).
- **Com pose de qualidade e bom treino**, a pose iguala ou supera o RGB ([SPOTER+MediaPipe](../literatura/bohacek-2022-spoter-mediapipe.md), [Uni-Sign](../literatura/li-2025-uni-sign.md), [De Coster 2023](../literatura/decoster-2023-sign-embeddings.md)).
- **Limitações conhecidas da pose:**
  - falha em contatos mão–mão e mão–rosto ([Moryossef 2021](../literatura/moryossef-2021-pose-estimation.md));
  - modelos de pose captam melhor a configuração de mão e o movimento, e modelos de pixel captam melhor o ponto de articulação ([Yin 2026](../literatura/yin-2026-phonological-perception.md)).

## 2.6 Normalização, aumento de dados e pontos ausentes

- **Normalização** é o fator de maior impacto ([De Coster 2023](../literatura/decoster-2023-sign-embeddings.md); [Zelezny 2025](../literatura/zelezny-2025-pose-ablation.md)). O padrão é normalizar o corpo pelos ombros, ou pelo espaço de sinalização, e cada mão pelo próprio punho.
- **Interpolação de pontos ausentes** tem efeito pequeno em benchmarks de ASL, mas grande em Libras (dos Santos 2025).
- **Rotação forte como aumento** prejudica o resultado.
- Não há estudo sistemático que varie subconjunto de pontos, normalização, imputação e aumentos em conjunto, com protocolo LOPO.

## 2.7 Lacunas e posicionamento

Lacunas que este projeto pode explorar, todas ligadas à meta de vocabulário completo:
1. Supervisão fonológica no reconhecimento de Libras. Exige anotar os parâmetros dos sinais.
2. Busca *one-shot* em dicionários de Libras, com encoder de pose pré-treinado em outras línguas.
3. Benchmark entre datasets, com landmarks e independente de sinalizador (treino em MINDS/UFOP, teste no MALTA-LIBRAS).
4. Pré-treino auto-supervisionado em pose com vídeos de Libras sem rótulo.
5. Ablação sistemática do pré-processamento com protocolo LOPO.
6. Ponto de articulação com landmarks: medir a perda em pares mínimos do LIBRAS-UFOP e testar *features* que relacionem a posição da mão com o rosto e o corpo.
