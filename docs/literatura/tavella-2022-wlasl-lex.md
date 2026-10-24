# WLASL-LEX: a Dataset for Recognising Phonological Properties in American Sign Language

- **Referência:** Federico Tavella, Viktor Schlegel, Marta Romeo, Aphrodite Galata, Angelo Cangelosi. ACL 2022 (artigo curto), pp. 453-463. https://aclanthology.org/2022.acl-short.49 · arXiv: https://arxiv.org/abs/2203.06096
- **Código / dados:** link e licença: a verificar (não constam na versão HTML consultada).
- **Problema:** A fonologia é pouco modelada em processamento de línguas de sinais. Os autores propõem reconhecer **propriedades fonológicas** em vez de glosas.
- **Método:**
  - Cruzamento por glosa entre o WLASL2000 e o ASL-LEX (1.0, segundo a versão consultada), resultando em **10.017 vídeos e 800 glosas**.
  - Seis propriedades: flexão, local maior, local menor, movimento, dedos selecionados e tipo de sinal.
  - Pose: FrankMocap (3D) e HRNet (2D), com subconjunto do tronco superior.
  - Modelos: maioria (baseline), MLP, RNN, I3D (RGB) e ST-GCN.
- **Dados e protocolo:**
  - Divisão 70:15:15 por propriedade, estratificada.
  - Dois cenários:
    - (a) *phoneme split*: as glosas podem aparecer no treino e no teste, "sinalizadas por pessoas diferentes";
    - (b) *gloss split*: as glosas de teste **não aparecem no treino**.
  - A independência de sinalizador no cenário (b) não é garantida explicitamente: a verificar.
- **Resultados principais** (acurácia, ST-GCN com HRNet, o melhor em todas as propriedades; fonte: arXiv 2203.06096):

  | Propriedade | *Phoneme split* | *Gloss split* (sinais não vistos) |
  |---|---|---|
  | Flexão | 62,3 | 49,1 |
  | Local maior | 83,2 | 77,3 |
  | Local menor | 74,5 | 55,1 |
  | Movimento | 63,6 | 52,5 |
  | Dedos selecionados | 73,8 | 65,7 |
  | Tipo de sinal | 84,5 | 76,6 |
- **Relevância para este projeto:**
  - Mostra que GCNs sobre esqueleto reconhecem parâmetros fonológicos **mesmo em sinais nunca vistos no treino**. É a base para um reconhecedor de vocabulário aberto por composição de parâmetros.
  - Indica quais parâmetros são mais fáceis (local maior, tipo de sinal) e quais são mais difíceis (flexão, movimento, local menor).
  - Útil para planejar a ordem de investigação na Libras.
- **Limitações / críticas:**
  - Reconhecer cada parâmetro com 50–77% ainda não basta para identificar o sinal.
  - Faltam configuração de mão completa e orientação como classes.
  - O ganho do 3D (FrankMocap) não foi consistente.
