# Índice dos fichamentos (Fase 1 – revisão do estado da arte)

Revisão feita em 2026-10-22. Cada fichamento traz a fonte primária (DOI ou URL), o protocolo de avaliação e os pontos marcados como "a verificar". Os números de trabalhos diferentes **só são comparáveis** quando o protocolo (dependente ou independente de sinalizador) e o dataset coincidem.

## 1. ISLR baseado em esqueleto/landmarks

| Fichamento | Ideia central | Protocolo |
|---|---|---|
| [jiang-2021-sam-slr](jiang-2021-sam-slr.md) | SL-GCN (27 nós, 4 fluxos) + *ensemble* multimodal; 1º no ChaLearn 2021 (AUTSL) | Independente (AUTSL) |
| [selvaraj-2022-openhands](selvaraj-2022-openhands.md) | Biblioteca com LSTM, transformer, ST-GCN e SL-GCN sobre MediaPipe (27 pontos), em 7 datasets; pré-treino DPC | Divisões oficiais (varia) |
| [bohacek-2022-spoter](bohacek-2022-spoter.md) | Transformer leve sobre pose; normalização corpo + mão local; aumentos | Divisões oficiais (WLASL) |
| [bohacek-2022-spoter-mediapipe](bohacek-2022-spoter-mediapipe.md) | Trocar só o estimador para MediaPipe: 63,18 → 78,29% no WLASL100, acima do I3D | Divisões oficiais (WLASL) |
| [hu-2023-signbert-plus](hu-2023-signbert-plus.md) | Pré-treino auto-supervisionado com mascaramento de pose da mão e *prior* MANO | MSASL (independente), WLASL (oficial) |
| [sohn-2023-kaggle-islr](sohn-2023-kaggle-islr.md) | 1º lugar no Kaggle ISLR: 1D CNN + transformer sobre MediaPipe (lábios + mãos), cerca de 1,8 M de parâmetros; notas sobre o 2º lugar | A verificar |
| [li-2025-uni-sign](li-2025-uni-sign.md) | Pré-treino gerativo em escala; só pose iguala RGB+pose (WLASL2000 63,1%) | MSASL (independente), WLASL (oficial) |

## 2. Trabalhos específicos de Libras

| Fichamento | Método | Datasets | Protocolo | Resultado principal |
|---|---|---|---|---|
| [alves-2024-skeleton-image](alves-2024-skeleton-image.md) | OpenPose → imagem de esqueleto → ResNet-18 | MINDS-Libras, LIBRAS-UFOP | *Nested* LOPO | 0,93 / 0,82 (acurácia) |
| [santos-2025-landmark-subset](santos-2025-landmark-subset.md) | MediaPipe + subconjunto de pontos + imputação por *spline* | MINDS-Libras, LIBRAS-UFOP | *Nested* LOPO | 0,94 / 0,91 (acurácia) |
| [naz-2024-mse-gcn](naz-2024-mse-gcn.md) | GCN multiescala + atenção ST-JPA | WLASL, MINDS-Libras, LIBRAS-UFOP | A verificar ("menos robusto que LOPO" segundo dos Santos) | 97,44% / 88,59% |
| [costa-2025-i3d-libras](costa-2025-i3d-libras.md) | I3D RGB, tempo real | MINDS-Libras | Independente (detalhes a verificar) | 92,5% |
| [costa-2026-distilacao-mamba](costa-2026-distilacao-mamba.md) | Distilação I3D → Mamba 1D sobre MediaPipe | Libras (dataset a verificar) | Independente | 71% → 78% |
| [delucis-2025-malta-libras](delucis-2025-malta-libras.md) | VideoMAE, TimeSformer, ViViT; novo conjunto MALTA-LIBRAS (PUCRS) | MINDS → MALTA-LIBRAS | Entre datasets | Qualitativo (números a verificar) |

**Não fichados** (texto completo inacessível ou números só por fonte secundária; citados dentro dos fichamentos acima):
- Passos et al. 2021, GEI + DensePose, IEEE TCAS-I, DOI 10.1109/TCSI.2021.3091001, código https://github.com/wesleylp/libras.
- De Castro et al. 2023, CNN 3D multi-fluxo com profundidade artificial, ESWA 215:119394, DOI 10.1016/j.eswa.2022.119394.
- Arcanjo et al. 2024, FastDTW sobre ângulos MediaPipe, WebMedia, DOI 10.5753/webmedia.2024.243245.
- Lazzarotto et al. 2025, CNNs 3D MINDS → MALTA, ERAMIA-RS, DOI 10.5753/eramiars.2025.16773.
- Os artigos dos datasets (Rezende et al. 2021, MINDS-Libras, DOI 10.1007/s00521-021-05802-4; Cerna et al. 2021, LIBRAS-UFOP, DOI 10.1016/j.eswa.2020.114179; V-LIBRASIL) ficam com o agente de datasets em `docs/datasets/`.

## 3. Escalar para vocabulário grande (fonologia, pré-treino, few/zero/one-shot)

| Fichamento | Ideia central | Protocolo |
|---|---|---|
| [tavella-2022-wlasl-lex](tavella-2022-wlasl-lex.md) | Reconhecer 6 propriedades fonológicas com ST-GCN, inclusive em sinais não vistos | Disjunto em glosa (sinalizador a verificar) |
| [kezar-2023-phonology](kezar-2023-phonology.md) | Cabeças fonológicas auxiliares: SL-GCN de 29,4 → 38,1% no WLASL2000 | Divisão oficial do WLASL |
| [kezar-2023-sem-lex](kezar-2023-sem-lex.md) | 84 mil vídeos, 3.149 sinais, rótulos fonológicos; +6 pontos com 4 exemplos por sinal | Independente |
| [bilge-2022-zero-shot](bilge-2022-zero-shot.md) | *Zero-shot* com descrições de dicionário e atributos fonológicos (RGB) | Disjunto em classe |
| [vandendriessche-2025-one-shot](vandendriessche-2025-one-shot.md) | Encoder de pose congelado + busca em dicionário com 1 vídeo por sinal (10.235 sinais, Rec@1 0,374) | Sinalizador a verificar |
| [hu-2023-signbert-plus](hu-2023-signbert-plus.md), [selvaraj-2022-openhands](selvaraj-2022-openhands.md), [li-2025-uni-sign](li-2025-uni-sign.md) | Pré-treino auto-supervisionado/gerativo em pose (ver tema 1) | — |

## 4. RGB vs. pose

| Fichamento | Achado |
|---|---|
| [li-2020-wlasl](li-2020-wlasl.md) | Com OpenPose, o I3D vence a pose no top-1 (32,48 contra 23,65 no WLASL2000) |
| [moryossef-2021-pose-estimation](moryossef-2021-pose-estimation.md) | Falhas da pose em contato mão–mão e mão–rosto; acertos têm mais quadros com mão detectada |
| [desai-2023-asl-citizen](desai-2023-asl-citizen.md) | Independente de sinalizador, 2.731 sinais: I3D 63,10 contra ST-GCN/MediaPipe 59,52 (Rec@1) |
| [yin-2026-phonological-perception](yin-2026-phonological-perception.md) | Pose capta melhor configuração de mão e movimento; pixel capta melhor local |
| [bohacek-2022-spoter-mediapipe](bohacek-2022-spoter-mediapipe.md), [li-2025-uni-sign](li-2025-uni-sign.md), [vandendriessche-2025-one-shot](vandendriessche-2025-one-shot.md), [decoster-2023-sign-embeddings](decoster-2023-sign-embeddings.md) | Pose bem feita iguala ou supera RGB |
| [costa-2025-i3d-libras](costa-2025-i3d-libras.md), [costa-2026-distilacao-mamba](costa-2026-distilacao-mamba.md) | Em Libras: RGB > landmarks no mesmo protocolo; a distilação reduz a diferença |

## 5. Normalização, aumento de dados e pontos ausentes

| Fichamento | Achado |
|---|---|
| [decoster-2023-sign-embeddings](decoster-2023-sign-embeddings.md) | Normalização corpo (ombros) + mão (punho) é crucial; imputação linear com efeito pequeno no MediaPipe |
| [santos-2025-landmark-subset](santos-2025-landmark-subset.md) | Em Libras: subconjunto de pontos (todos os 543 é muito pior) e imputação por *spline* (+4 a +18 pontos de F1) |
| [zelezny-2025-pose-ablation](zelezny-2025-pose-ablation.md) | Normalização SignSpace domina; interpolação curta ajuda pouco; rotação forte prejudica |
| [bohacek-2022-spoter](bohacek-2022-spoter.md) | Normalização pelo espaço de sinalização + mão local; aumento por rotação sequencial de articulações |
| [sohn-2023-kaggle-islr](sohn-2023-kaggle-islr.md) | Receita completa: centro e escala com NaN ignorado, velocidade e aceleração, *flip* com troca de mãos, reamostragem temporal, mascaramento |
