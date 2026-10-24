# Combining Efficient and Precise Sign Language Recognition: Good pose estimation library is all you need

- **Referência:** Matyáš Boháček, Zhuo Cao, Marek Hrúz. CVPR 2022 Workshops (AVA). arXiv: https://arxiv.org/abs/2210.00893
- **Código:** o artigo cita uma demo em Gradio no Hugging Face Spaces (demo.signlanguagerecognition.com) e diz que ela é aberta, mas não dá o link do repositório: a verificar. O código-base é o do SPOTER (https://github.com/matyasbohacek/spoter).
- **Problema:** Medir quanto do desempenho de um modelo de pose depende do estimador de pose.
- **Método:**
  - SPOTER inalterado, com a mesma configuração do artigo original. A única troca é o estimador: Vision API da Apple → MediaPipe (BlazePose + mãos).
  - Os pontos do MediaPipe são reduzidos a 54 (5 da cabeça e 21 por mão), e o pescoço é calculado como ponto médio dos ombros.
  - Os parâmetros dos aumentos foram ajustados por busca de hiperparâmetros (W&B).
- **Dados e protocolo:** Apenas o WLASL100, com as divisões oficiais de treino, validação e teste. A independência de sinalizador não é explícita no WLASL.
- **Resultados principais** (WLASL100, top-1; fonte: arXiv 2210.00893):

  | Modelo | Entrada | Top-1 |
  |---|---|---|
  | SPOTER + MediaPipe | pose | **78,29%** |
  | TK-3D ConvNet | RGB | 77,55% |
  | Fusion-3 | RGB | 75,67% |
  | I3D | RGB | 65,89% |
  | SPOTER + Vision API (original) | pose | 63,18% |
  | Pose-TGCN | pose | 55,43% |

  Os autores reportam que o SPOTER tem cerca de metade dos parâmetros do I3D e inferência cerca de 11 vezes mais rápida.
- **Relevância para este projeto:**
  - Forte evidência a favor do ADR-0002: com MediaPipe, um transformer leve sobre pose supera métodos RGB no WLASL100.
  - Mostra que a **qualidade do estimador** domina o resultado: +15 pontos só trocando o estimador.
  - Reforça a consequência do ADR-0002 de medir a qualidade da extração na Fase 2.
- **Limitações / críticas:**
  - Avalia só o WLASL100, sem protocolo explicitamente independente de sinalizador.
  - Não há análise de erro por tipo de sinal.
  - A comparação com RGB usa números de terceiros.
