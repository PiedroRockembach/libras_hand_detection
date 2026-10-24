# OpenHands: Making Sign Language Recognition Accessible with Pose-based Pretrained Models across Languages

- **Referência:** Prem Selvaraj, Gokul NC, Pratyush Kumar, Mitesh Khapra. ACL 2022 (long), pp. 2114-2133. https://aclanthology.org/2022.acl-long.150 · arXiv: https://arxiv.org/abs/2110.05877
- **Código:** https://github.com/AI4Bharat/OpenHands (**PyTorch Lightning**, orientado a configuração). Os autores dizem liberar dados de pose e modelos "sob licenças permissivas"; a licença exata está a verificar.
- **Problema:** Padronizar o ISLR baseado em pose entre várias línguas de sinais e usar pré-treino auto-supervisionado para cenários com poucos dados.
- **Método:**
  - Pose do **MediaPipe Holistic** (modelo *heavy*), reduzida a **27 pontos 2D** esparsos (tronco, mãos e face).
  - Quatro modelos:
    - LSTM bidirecional (4 camadas, com atenção temporal);
    - transformer do tipo BERT (5 camadas, 6 cabeças, dimensão 128);
    - ST-GCN (10 camadas);
    - SL-GCN (10 blocos).
  - **Normalização:** pelas dimensões do vídeo e por centro e escala. Quadros ausentes ou corrompidos são interpolados.
  - **Aumentos:** cisalhamento, rotação (±π/3), escala, deslocamento temporal aleatório e subamostragem temporal uniforme e aleatória.
  - **Pré-treino:**
    - corpus de 1.129 h de vídeos de língua de sinais indiana (YouTube), com cerca de 100 M de quadros de pose;
    - foram testadas predição mascarada, aprendizado contrastivo e DPC (*Dense Predictive Coding*); só o DPC com encoder ST-GCN deu ganho.
- **Dados e protocolo:**
  - Datasets: AUTSL, CSL, WLASL2000, GSL, LSA64, DEVISIGN_L e INCLUDE.
  - Usa as divisões fornecidas por cada dataset (treino e teste). O artigo **não discute** independência de sinalizador; ela depende de cada dataset (no AUTSL é independente).
- **Resultados principais** (teste, top-1; fonte: arXiv 2110.05877):

  | Dataset | LSTM | Transformer | ST-GCN | SL-GCN |
  |---|---|---|---|---|
  | AUTSL | 77,4 | 81,0 | 90,4 | 91,9 |
  | WLASL2000 | 20,6 | 23,2 | 21,4 | 30,6 |
  | INCLUDE | 83,0 | 90,4 | 91,2 | 93,5 |
  | LSA64 | 90,2 | 92,5 | 94,7 | 97,8 |

  - O SL-GCN com MediaPipe chega a 91,9% no AUTSL. O SL-GCN com HRNet do SAM-SLR chegava a 95,02% (os autores atribuem a diferença à pose de maior qualidade).
  - **Custo:** o LSTM é cerca de 10 vezes mais rápido que o SL-GCN em CPU, com cerca de 10% a menos de acurácia; o ST-GCN é cerca de 2 vezes mais rápido que o SL-GCN, com cerca de 3% a menos.
  - **DPC:**
    - INCLUDE: 91,2 → 94,7.
    - Com poucos exemplos por classe no INCLUDE: 3/classe 15,2 → 35,4; 5/classe 45,0 → 57,4; 10/classe 79,7 → 86,3.
    - LSA64 com 3/classe: 39,7 → 57,2.
    - WLASL2000 com 3/classe: 1,6 → 2,8 (quase nulo).
  - **Transferência entre línguas:** o pré-treino em língua de sinais indiana melhorou o ajuste fino em ASL, CSL e LSA em cerca de 6, 4 e 2 pontos.
- **Relevância para este projeto:**
  - Biblioteca e protocolo mais próximos do nosso pipeline (MediaPipe → GCN/transformer).
  - Dá uma comparação controlada entre LSTM, transformer, ST-GCN e SL-GCN com a mesma entrada. Os GCNs vencem na maioria dos datasets, o que justifica a Fase 4.
  - O pré-treino em pose ajuda muito com 3 a 10 exemplos por classe, que é exatamente o regime da Libras.
  - A seleção de 27 pontos é reutilizada pelo ASL Citizen e pelo Sem-Lex.
- **Limitações / críticas:**
  - No WLASL2000 tudo fica abaixo de 31%; o pré-treino não resolve vocabulário grande com poucos exemplos.
  - A predição mascarada não funcionou aqui, ao contrário do SignBERT+, provavelmente por diferenças de projeto.
  - Usa só 2D e 27 pontos, sem análise da face.
