# Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison (WLASL)

- **Referência:** Dongxu Li, Cristian Rodriguez Opazo, Xin Yu, Hongdong Li. WACV 2020. arXiv: https://arxiv.org/abs/1910.11006
- **Código / dados:** https://dxli94.github.io/WLASL/ (dataset e baselines). A licença do dataset é para o agente de datasets verificar.
- **Problema:** Criar um benchmark de ISLR com vocabulário grande e comparar modelos de aparência (RGB) com modelos de pose.
- **Método:**
  - RGB: VGG-GRU e I3D (pré-treinado em ImageNet/Kinetics).
  - Pose: Pose-GRU e **Pose-TGCN** (GCN temporal) sobre 55 pontos 2D do OpenPose (13 do tronco superior e 21 por mão).
- **Dados e protocolo:**
  - WLASL2000: 2.000 glosas, 21.083 vídeos, 119 sinalizadores, cerca de 10,5 amostras por glosa.
  - Subconjuntos WLASL100, WLASL300 e WLASL1000.
  - Divisão 4:1:1 **por glosa** (amostras de cada glosa repartidas entre treino, validação e teste). O artigo **não garante** que os sinalizadores sejam disjuntos entre partições, embora remova repetições do mesmo sinalizador.
  - Na prática, **não é um protocolo independente de sinalizador estrito**.
- **Resultados principais** (top-1 / top-5 / top-10; fonte: arXiv 1910.11006):

  | Método | WLASL100 | WLASL2000 |
  |---|---|---|
  | Pose-GRU | 46,51 / 76,74 / 85,66 | 22,54 / 49,81 / 61,38 |
  | Pose-TGCN | 55,43 / 78,68 / 87,60 | 23,65 / 51,75 / 62,24 |
  | VGG-GRU | 25,97 / 55,04 / 63,95 | 8,44 / 23,58 / 32,58 |
  | I3D | **65,89** / 84,11 / 89,92 | **32,48** / 57,31 / 66,31 |
- **Relevância para este projeto:**
  - Primeira comparação direta RGB × pose em vocabulário grande. Com OpenPose (2019), o I3D vence no top-1 por cerca de 9–10 pontos, mas a pose empata no top-5 e no top-10.
  - Esse resultado foi depois revertido com estimadores melhores (bohacek-2022-spoter-mediapipe: 78,29% só com pose no WLASL100) e com pré-treino (li-2025-uni-sign).
  - Benchmark padrão para comparar o nosso modelo fora da Libras.
- **Limitações / críticas:**
  - O protocolo não é estritamente independente de sinalizador, logo não é comparável a LOSO.
  - Vídeos da web com qualidade heterogênea.
  - Parte dos vídeos não está mais disponível (a verificar com o agente de datasets).
