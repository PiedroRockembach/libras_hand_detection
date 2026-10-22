# Uni-Sign: Toward Unified Sign Language Understanding at Scale

- **Referência:** Zecheng Li, Wengang Zhou, Weichao Zhao, Kepeng Wu, Hezhen Hu, Houqiang Li. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html · arXiv: https://arxiv.org/abs/2501.15187
- **Código:** https://github.com/ZechengLi19/Uni-Sign (dados e código). O framework é provavelmente PyTorch, mas está a verificar.
- **Problema:** Reduzir a distância entre o pré-treino e o ajuste fino em compreensão de língua de sinais (ISLR, CSLR e tradução) por meio de pré-treino gerativo em grande escala.
- **Método:**
  - Pose do RTMPose-x (MMPose, 133 pontos), dos quais são usados 69: 21 por mão, 9 do corpo e 18 da face.
  - Encoders temporais ST-GCN de 3 camadas, EfficientNet-B0 para RGB e mT5-Base como modelo de linguagem.
  - Três estágios: pré-treino só com pose → pré-treino contínuo RGB+pose → ajuste fino.
  - Todas as tarefas, incluindo o ISLR, são tratadas como **tradução** (gerar o texto da glosa).
  - A fusão RGB-pose é guiada por *prior* (PGF), com amostragem ciente da confiança dos pontos.
  - Pré-treino: CSL-News (1.985 h, 751.320 clipes) e YouTube-ASL (984 h) para a ASL.
- **Dados e protocolo:**
  - MSASL100/1000: o MS-ASL declara suporte a teste independente de sinalizador (https://arxiv.org/abs/1812.01053).
  - WLASL100/2000: divisão oficial; a independência de sinalizador não é explícita.
  - Reporta acurácias por instância (P-I) e por classe (P-C).
- **Resultados principais** (top-1 P-I / P-C; fonte: arXiv 2501.15187, Tabela 3):

  | Benchmark | SignBERT+ | NLA-SLR (pose+RGB) | Uni-Sign só pose | Uni-Sign pose+RGB |
  |---|---|---|---|---|
  | MSASL1000 | 62,42 / 60,15 | 72,56 / 69,86 | 77,88 / 76,55 | 78,16 / 76,97 |
  | WLASL2000 | 48,85 / 46,37 | 61,05 / 58,05 | 63,13 / 60,90 | 63,52 / 61,32 |
  | WLASL100 | 79,84 / 80,72 | 91,47 / 92,17 | 92,24 / 92,75 | 92,25 / 92,67 |

  - No MSASL1000 com pose, o ajuste fino específico da tarefa dá 56,92 (P-I), contra 77,88 no paradigma unificado (Tabela 7).
- **Relevância para este projeto:**
  - Evidência atual (2025) de que, **com pré-treino grande, só a pose alcança e supera RGB+pose**: o RGB acrescenta menos de 1 ponto. Isso apoia o ADR-0002.
  - Mostra que o gargalo para vocabulário grande é o **pré-treino em escala**, não a modalidade.
  - Tratar o ISLR como geração de texto abre caminho para vocabulário aberto.
- **Limitações / críticas:**
  - Exige corpora de milhares de horas com legenda, que não existem para a Libras (é uma lacuna).
  - O modelo é grande (mT5), incompatível com tempo real em notebook.
  - Usa RTMPose, não MediaPipe.
  - Não isola o efeito do pré-treino no ISLR de forma limpa.
