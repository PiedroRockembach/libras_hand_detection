# SignBERT+: Hand-model-aware Self-supervised Pre-training for Sign Language Understanding

- **Referência:** Hezhen Hu, Weichao Zhao, Wengang Zhou, Houqiang Li. IEEE TPAMI, 2023 (segundo a página do autor na USTC, https://faculty.ustc.edu.cn/zhouwengang/en/lwcg/53276/list/index.htm, e a IEEE CSDL, https://computer.org/csdl/journal/tp/2023/09/10109128/1MESMCioCQM). DOI: a verificar. arXiv: https://arxiv.org/abs/2305.04868 . Extensão do SignBERT (ICCV 2021, https://arxiv.org/abs/2110.05382).
- **Código:** a verificar (não encontrei repositório oficial).
- **Problema:** Há pouco dado rotulado para compreensão de língua de sinais. A proposta é usar pré-treino auto-supervisionado sobre pose das mãos.
- **Método:**
  - Entrada: pontos 2D do MMPose (133 de corpo inteiro), usando 21 por mão mais as articulações do braço para posição global.
  - Cada pose de mão por quadro vira um *token* visual, processado por um encoder do tipo transformer.
  - O pré-treino mascara em três níveis: articulação (zerar ou perturbar), quadro inteiro e clipe (2 a K quadros consecutivos; K = 8 foi o melhor).
  - O decodificador é **consciente do modelo de mão (MANO)**: regride pose (θ), forma (β) e câmera e reprojeta as articulações em 2D. Isso funciona como *prior* anatômico.
  - A perda de reconstrução é ponderada pela confiança da detecção.
  - Dados de pré-treino: cerca de 230.246 vídeos, juntando os treinos dos datasets de sinais e fontes adicionais.
- **Dados e protocolo:**
  - MSASL (1.000 sinais, 25.512 amostras): segundo os autores do MS-ASL (https://arxiv.org/abs/1812.01053), o dataset permite teste independente de sinalizador.
  - WLASL (2.000 sinais, 21.083 amostras): divisão oficial; a independência de sinalizador não é explícita (ver li-2020-wlasl).
  - SLR500 (500 sinais; 90.000 amostras de treino e 35.000 de teste).
  - São reportadas acurácias por instância (P-I) e por classe (P-C).
- **Resultados principais** (top-1; fonte: arXiv 2305.04868):

  | Dataset | Métrica | Só pose | Pose + RGB |
  |---|---|---|---|
  | MSASL1000 | P-I | 62,42% | 73,71% |
  | MSASL1000 | P-C | 60,15% | 70,77% |
  | WLASL2000 | P-I | 48,85% | 55,59% |
  | WLASL2000 | P-C | 46,37% | 53,33% |
  | SLR500 | acurácia | 95,4% | 97,8% |
- **Relevância para este projeto:**
  - Principal referência de pré-treino auto-supervisionado em pose. A ideia de mascarar articulações, quadros e clipes se aplica diretamente a landmarks MediaPipe.
  - Esse pré-treino pode aproveitar vídeos de Libras sem rótulo (dicionários, vídeos online).
  - O *prior* de mão é uma forma de embutir conhecimento anatômico, útil para a configuração de mão.
- **Limitações / críticas:**
  - Nos benchmarks grandes, o ganho do RGB sobre a pose é alto (cerca de 11 pontos no MSASL1000), o que contraria em parte o ADR-0002. Trabalhos posteriores (Uni-Sign, li-2025-uni-sign) reduzem essa diferença com pose melhor e pré-treino maior.
  - O pré-treino exige centenas de milhares de vídeos.
  - Sem código oficial confirmado.
