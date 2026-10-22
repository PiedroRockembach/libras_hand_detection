# ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition

- **Referência:** Aashaka Desai et al. NeurIPS 2023 Datasets and Benchmarks Track. https://papers.neurips.cc/paper_files/paper/2023/hash/f29cf8f8b4996a4a453ef366cf496354-Abstract-Datasets_and_Benchmarks.html · arXiv: https://arxiv.org/abs/2304.05934 · projeto: https://www.microsoft.com/en-us/research/project/asl-citizen/
- **Código:** o artigo diz que código e pesos são liberados junto com o dataset, a partir da página do projeto. Repositório e framework específicos: a verificar. A licença do dataset é para o agente de datasets verificar.
- **Problema:** Os datasets de ISLR têm poucos sinalizadores e vídeos da web sem consentimento. Os autores propõem um dataset coletado com a comunidade e enquadram o ISLR como **busca em dicionário**.
- **Método:**
  - Baselines: **I3D** (RGB) e **ST-GCN** (pose) sobre **MediaPipe Holistic**, com o subconjunto esparso de 27 pontos do OpenHands.
  - Métricas de recuperação: Rec@k, MRR e DCG.
- **Dados e protocolo:**
  - 83.399 vídeos, 2.731 sinais, 52 sinalizadores, gravados por webcam em ambientes variados.
  - Divisão: 35 sinalizadores no treino, 6 na validação e 11 no teste. **Independente de sinalizador** (o teste só tem sinalizadores fora do treino e da validação).
- **Resultados principais** (teste do ASL Citizen; fonte: arXiv 2304.05934):

  | Modelo | Rec@1 | Rec@5 | Rec@10 | MRR |
  |---|---|---|---|---|
  | I3D (RGB) | **63,10%** | 86,09% | 90,86% | 0,7332 |
  | ST-GCN (pose, MediaPipe) | 59,52% | 82,68% | 88,13% | 0,6997 |

  - Num subconjunto de 1.075 glosas em comum com o WLASL: o modelo treinado no WLASL acerta 8,49% no teste do ASL Citizen, contra 74,16% do modelo treinado no ASL Citizen. A mudança de domínio é enorme.
- **Relevância para este projeto:**
  - Comparação **direta e controlada** RGB × pose com vocabulário grande (2.731) e protocolo independente de sinalizador. O RGB vence por cerca de 3,6 pontos com baselines simples.
  - Vandendriessche et al. 2025 depois chegam a Rec@1 de 0,751 só com pose MediaPipe, acima do I3D.
  - Modelo de protocolo para o nosso caso de uso de dicionário: métricas de recuperação e teste com sinalizadores novos.
  - Candidato a fonte de pré-treino em pose (ver vandendriessche-2025-one-shot).
- **Limitações / críticas:**
  - Só ASL.
  - Os baselines não usam normalização nem aumentos elaborados.
  - A coleta por webcam tem qualidade variável, o que é bom para robustez.
