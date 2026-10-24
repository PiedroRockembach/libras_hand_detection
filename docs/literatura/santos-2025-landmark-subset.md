# Proper Body Landmark Subset Enables More Accurate and 5X Faster Recognition of Isolated Signs in LIBRAS

- **Referência:** Daniele L. V. dos Santos, Thiago B. Pereira, Carlos Eduardo G. R. Alves, Richard J. M. G. Tello, Francisco de A. Boldt, Thiago M. Paixão. arXiv 2510.24887 (v1 de 28/10/2025, v4 de 17/04/2026); aceito no IEEE SAS 2026 segundo os comentários do arXiv. https://arxiv.org/abs/2510.24887
- **Código:** https://github.com/danielelvs/islr-subset (citado no artigo). Framework e licença: a verificar.
- **Problema:** Trocar o OpenPose pelo MediaPipe acelera o pipeline mas derruba a acurácia em Libras. A pergunta é como recuperar o desempenho.
- **Método:**
  - Mesmo pipeline de Alves et al. 2024 (Skeleton-DML → imagem 224×224 → ResNet-18), mas com landmarks **MediaPipe**.
  - Subconjuntos de pontos comparados:
    - todos (543);
    - "Laines" (68);
    - "Arcanjo" (75: pose + mãos);
    - "ASL-1st" (118: face + mãos, do 1º lugar no Kaggle);
    - "ASL-2nd" (80: lábios + mãos + pose, do 2º lugar).
  - **Imputação** de pontos ausentes por *spline* por partes, por coordenada, em janela de 5 quadros. Usa spline cúbica quando há pelo menos 4 pontos válidos e linear nos demais casos.
- **Dados e protocolo:**
  - MINDS-Libras (1.155 vídeos, 20 sinais, 12 sinalizadores) e LIBRAS-UFOP (3.040 vídeos, 56 sinais, 5 sinalizadores).
  - ***Nested* LOPO**, independente de sinalizador.
- **Resultados principais** (F1, com imputação; fonte: arXiv 2510.24887):

  | Subconjunto | MINDS-Libras | LIBRAS-UFOP |
  |---|---|---|
  | Todos (543) | 0,66 | 0,69 |
  | Laines (68) | 0,94 | 0,87 |
  | Arcanjo (75) | 0,94 | 0,90 |
  | ASL-1st (118) | 0,94 | 0,86 |
  | ASL-2nd (80) | **0,94** | **0,91** |

  - Acurácia do ASL-2nd: 0,94 (MINDS-Libras) e 0,91 (LIBRAS-UFOP).
  - A imputação dá ganho de F1 de cerca de 4–6 pontos no MINDS e 5–18 no UFOP, conforme o subconjunto.
  - O pipeline fica 5,4–5,7 vezes mais rápido que o de Alves et al. com OpenPose.
  - Comparação com o estado da arte:
    - Alves et al. 2024 (LOPO): 0,94 / 0,82.
    - De Castro et al. 2023 (LOPO): 0,91 / 0,74.
    - Naz et al. (MSE-GCN): 0,97 no MINDS, com protocolo "menos robusto que LOPO" segundo os autores; o protocolo no UFOP é descrito de forma inconsistente entre tabela e texto.
- **Relevância para este projeto:**
  - É o trabalho de Libras mais próximo do nosso pipeline: MediaPipe, protocolo LOPO e os mesmos datasets candidatos.
  - Dois achados diretos para os ADRs:
    - (1) usar **todos** os 543 pontos é muito pior (F1 0,66): a face completa dilui o sinal. É preciso escolher um subconjunto.
    - (2) a **imputação temporal** de pontos ausentes ajuda muito.
  - Isso apoia a máscara prevista no ADR-0002, mas sugere testar imputação além do mascaramento.
- **Limitações / críticas:**
  - O classificador é uma CNN sobre imagem de esqueleto, não um modelo temporal.
  - Vocabulário pequeno.
  - Não compara imputação com mascaramento aprendido.
  - A comparação com Naz et al. tem protocolo inconsistente, como os próprios autores reconhecem.
