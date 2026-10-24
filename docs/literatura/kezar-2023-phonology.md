# Improving Sign Recognition with Phonology

- **Referência:** Lee Kezar, Jesse Thomason, Zed Sehyr. EACL 2023, pp. 2732-2737. https://aclanthology.org/2023.eacl-main.200 · arXiv: https://arxiv.org/abs/2302.05759
- **Código:** https://github.com/leekezar/ImprovingSignRecognitionWithPhonology. Os modelos vêm do OpenHands (PyTorch Lightning).
- **Problema:** Modelos de ISLR (por exemplo, o SL-GCN) às vezes preveem sinais visual e fonologicamente sem relação com o correto. A pergunta é se supervisionar parâmetros fonológicos ajuda.
- **Método:**
  - Aprendizado **multitarefa** sobre pose: além da glosa, cabeças auxiliares preveem atributos fonológicos do ASL-LEX 2.0 (16 tipos, como configuração de mão com 49 classes, local menor com 37 e movimento de trajetória com 8).
  - Os ganhos vêm principalmente das cabeças de **configuração de mão** e **local menor**.
  - Modelos: SL-GCN e transformer, ambos do OpenHands.
- **Dados e protocolo:**
  - WLASL2000 (21.083 vídeos), com divisões oficiais.
  - 754 das 2.000 glosas têm rótulo fonológico (9.438 vídeos, cerca de 48%).
  - A independência de sinalizador **não é declarada** (é a divisão oficial do WLASL).
  - Média de 4 *seeds*.
- **Resultados principais** (WLASL2000, teste; fonte: arXiv 2302.05759):

  | Modelo | Top-1 sem → com fonologia | Top-3 sem → com fonologia | MRR |
  |---|---|---|---|
  | SL-GCN | 29,4 → **38,1** (+8,7) | 50,2 → 61,0 | 0,43 → 0,52 |
  | Transformer | 20,5 → 23,4 (+2,8) | 36,9 → 41,7 | 0,32 → 0,36 |
- **Relevância para este projeto:**
  - Evidência central para a direção do ADR-0001 (decomposição fonológica): supervisão auxiliar de parâmetros dá +8,7 pontos com vocabulário de 2.000 sinais, **sem mudar a arquitetura**.
  - Implementação simples: cabeças extras e soma de perdas.
  - Para a Libras, exige um léxico fonologicamente anotado, o equivalente a um "LIBRAS-LEX". Ver lacunas no índice.
- **Limitações / críticas:**
  - O protocolo não é explicitamente independente de sinalizador.
  - Só cerca de 48% dos vídeos têm rótulo fonológico.
  - O ganho depende da arquitetura (grande no GCN, pequeno no transformer).
