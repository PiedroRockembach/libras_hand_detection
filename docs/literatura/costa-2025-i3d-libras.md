# A Lightweight I3D-Based Approach for Real-Time Brazilian Sign Language Recognition

- **Referência:** Victor Costa, Leandro Tavares, Ruhan Conceição, Luciano Agostini, Brenda Salenave Santana, Tatiana Bolivar Lebedeff, Guilherme Corrêa (UFPel). WebMedia 2025 (artigo curto), pp. 501-505. DOI: 10.5753/webmedia.2025.16072 · https://sol.sbc.org.br/index.php/webmedia/article/view/37997
- **Código:** a verificar.
- **Problema:** ISLR em Libras em tempo real direto do vídeo RGB, sem extrair esqueleto.
- **Método:** I3D pré-treinado com ajuste fino em vídeos RGB. Detalhes de pré-treino e de entrada: a verificar (o PDF não foi legível pela ferramenta).
- **Dados e protocolo:**
  - MINDS-Libras.
  - **Independente de sinalizador**: os sujeitos de teste são "completamente não vistos no treino", segundo o resumo. Número de dobras e quais sinalizadores ficam no teste: a verificar.
- **Resultados principais:** acurácia de **92,5%** no MINDS-Libras (resumo), com inferência em tempo real; FPS a verificar.
- **Relevância para este projeto:**
  - Resultado RGB independente de sinalizador no mesmo dataset em que trabalhos de landmarks chegam a 0,93–0,94 em LOPO (Alves 2024; dos Santos 2025).
  - Em Libras, com vocabulário pequeno, RGB e pose ficam empatados (comparação indireta, com protocolos possivelmente diferentes).
  - Base para o trabalho de distilação do mesmo grupo (costa-2026-distilacao-mamba).
- **Limitações / críticas:**
  - Artigo curto, só com o resumo verificado.
  - O protocolo exato (LOPO ou divisão fixa) é desconhecido, então a comparação com LOPO é aproximada.
