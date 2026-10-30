# 4. Experimentos e resultados

Cada experimento entra nesta tabela com link para a configuração e o identificador da execução. Números só entram aqui se vierem de um log de experimento (`LIBRAS_DATA_ROOT/experiments/<nome>/<execução>/summary.json`).

| ID | Data | Dataset / divisão | Modelo | Entrada | Top-1 | F1-macro | Config | Commit |
|---|---|---|---|---|---|---|---|---|
| `a0_stats_mlp_minds/20261030T175856Z-1b97ce9` | 2026-10-30 | MINDS Canon, LOSO 12 dobras | MLP sobre estatísticas temporais (531 mil parâmetros) | landmarks `mp_v2`, 64 quadros × 446 | 0,933 ± 0,052 | 0,920 ± 0,066 | [a0](../../configs/experiments/a0_stats_mlp_minds.yaml) | `1b97ce9` |
| `a1_gru_minds/20261030T181123Z-1b97ce9` | 2026-10-30 | MINDS Canon, LOSO 12 dobras | GRU bidirecional 2 × 192 + cosseno (1,30 M parâmetros) | idem | 0,934 ± 0,044 | 0,919 ± 0,054 | [a1](../../configs/experiments/a1_gru_minds.yaml) | `1b97ce9` |

| `a2_gru_mirror/20261030T193009Z-9d003ba` | 2026-10-30 | idem | a1 + espelhamento (p = 0,5) | idem | 0,941 ± 0,035 | 0,930 ± 0,043 | [a2](../../configs/experiments/a2_gru_mirror.yaml) | `9d003ba` |
| `a3_gru_kinect/20261030T193032Z-9d003ba` | 2026-10-30 | idem; treino + vista Kinect | a1 | idem | 0,936 ± 0,069 | 0,920 ± 0,090 | [a3](../../configs/experiments/a3_gru_kinect.yaml) | `9d003ba` |
| `a4_gru_noface/20261030T194650Z-9d003ba` | 2026-10-30 | idem | a1 | sem pontos do rosto | 0,955 ± 0,035 | 0,947 ± 0,044 | [a4](../../configs/experiments/a4_gru_noface.yaml) | `9d003ba` |
| `a5_gru_strongaug/20261030T200050Z-9d003ba` | 2026-10-30 | idem | a1 + aumento mais forte | idem | 0,947 ± 0,037 | 0,933 ± 0,043 | [a5](../../configs/experiments/a5_gru_strongaug.yaml) | `9d003ba` |
| `a6_gru_mirror_kinect/20261030T201549Z-3813419` | 2026-10-30 | idem; treino + vista Kinect | a1 + espelhamento | idem | **0,963 ± 0,025** | **0,958 ± 0,030** | [a6](../../configs/experiments/a6_gru_mirror_kinect.yaml) | `3813419` |
Média ± desvio-padrão (populacional) entre as 12 dobras. Em cada dobra, o teste é um sinalizador e a época é escolhida pela acurácia num outro sinalizador de validação (ADR-0004). O F1-macro de cada dobra considera as classes presentes no teste ou previstas, porque os sinalizadores 03, 04 e 09 não têm alguns sinais.

## 4.1 Fase A: baseline de sinais isolados (MINDS)

**Teste fora de domínio (MALTA, 109 clipes distintos com os 20 sinais do MINDS), modelos das 12 dobras.** Reavaliação de 2026-10-30 (`python -m libras.train.evaluate`, commit `d27857b`, pastas `eval-*` dentro de cada execução), depois de remover clipes duplicados (ver abaixo):

| Modelo | Top-1 | Top-5 | F1-macro |
|---|---|---|---|
| a0 (MLP estatísticas) | 0,521 ± 0,039 | 0,778 | 0,499 |
| a1 (GRU) | 0,635 ± 0,034 | 0,791 | 0,616 |

Top-1 médio por subcoleção (a0 → a1): acessibilidade_2 (20 clipes) 0,671 → 0,675; acessibilidade_3 (13) 0,500 → 0,603; spread_the_sign (14) 0,482 → 0,565; ufsc (19) 0,487 → 0,596; ufsc_v2 (18) 0,426 → 0,620; ufv (6) 0,222 → 0,444; usp (8) 0,625 → 0,792; youtube (11) 0,629 → 0,773.

**Duplicatas no MALTA:** a subcoleção `youtube_v2` é uma cópia da `youtube` (1.183 dos 1.185 arquivos distintos são idênticos byte a byte), e a `youtube` repete clipes com nomes de acentuação corrompida. A avaliação original, com 124 clipes, contava essas cópias e dava 0,545 (a0) e 0,658 (a1) no `summary.json` de cada execução. Desde o commit `d27857b`, a seleção mantém um clipe por conteúdo (SHA-256 do arquivo). Entre `ufsc` e `ufsc_v2` não há arquivos idênticos.

**Leitura:**

- No MINDS, o modelo que ignora a ordem dos quadros (a0) empata com a GRU (a1). Os 20 sinais do MINDS se separam pelas posições médias. A acurácia está na faixa relatada por dos Santos 2025 (0,94 ± 0,04, MediaPipe), mas o protocolo dele é LOPO aninhado, então a comparação não é direta.
- **Fora de domínio, a GRU é melhor (0,635 contra 0,521).** A queda de ~0,93 para ~0,64 confirma que a acurácia no MINDS não mede robustez (Delucis 2025), e é este o número a melhorar antes da demo.
- **Reprodutibilidade:** as execuções a partir do commit limpo repetiram exatamente os números de uma execução anterior com o mesmo código ainda não commitado (determinismo em GPU com seed fixa).

## 4.2 Fase A: ablações para robustez fora de domínio

Cada ablação muda uma única coisa em relação ao a1 (mesma seed, mesmas dobras). O MALTA usa os 109 clipes distintos. Comparação pareada por dobra contra o a1 (reavaliação `eval-20261030T192817Z-d27857b`):

| Execução | Mudança | Top-1 LOSO | Δ LOSO (dobras melhores/piores) | Top-1 MALTA | Δ MALTA (dobras melhores/piores) |
|---|---|---|---|---|---|
| a1 | baseline | 0,934 ± 0,044 | — | 0,635 ± 0,034 | — |
| a2 | espelhamento | 0,941 ± 0,035 | +0,007 (5/5) | **0,705 ± 0,035** | **+0,070 (11/1)** |
| a3 | + vista Kinect no treino | 0,936 ± 0,069 | +0,002 (6/4) | 0,683 ± 0,033 | +0,048 (11/1) |
| a4 | sem rosto | 0,955 ± 0,035 | +0,021 (6/2) | 0,633 ± 0,014 | −0,002 (5/6) |
| a5 | aumento mais forte | 0,947 ± 0,037 | +0,013 (7/3) | 0,613 ± 0,034 | −0,022 (3/8) |
| **a6** | **espelhamento + vista Kinect** | **0,963 ± 0,025** | +0,029 (7/4) | **0,729 ± 0,025** | **+0,093 (12/0)** |

**Leitura:**

- **Espelhamento e vista Kinect melhoram a generalização** de forma consistente (11 de 12 dobras cada), quase sem efeito no MINDS. O espelhamento simula sinalizadores com a outra mão dominante; o Kinect traz outro ângulo e outra câmera.
- **Rosto:** retirar os pontos do rosto melhora um pouco o MINDS e não muda o MALTA. Os 20 sinais do MINDS talvez dependam pouco de expressões não-manuais (a verificar). O rosto continua na configuração, porque o vocabulário maior (EQT, frases) deve precisar dele.
- **Aumento geométrico mais forte** piora o MALTA, em linha com Zelezny 2025 (rotação forte prejudica).
- **A combinação (a6) soma os dois ganhos:** MALTA 0,729, melhor que o a1 em 12 de 12 dobras e que o a2 em 9 de 12 (+0,024); top-5 no MALTA 0,899. Por subcoleção: acessibilidade_2 0,754; acessibilidade_3 0,904; spread_the_sign 0,685; ufsc 0,737; ufsc_v2 0,671; ufv 0,361 (6 clipes); usp 0,844; youtube 0,727. **O a6 é o modelo candidato à demo v0.**
- O a3 tem uma dobra fraca (sinalizador 12: 0,730), a investigar.

**Cuidado metodológico:** estas escolhas foram feitas olhando o MALTA, que assim passa a funcionar também como conjunto de seleção. Para um número fora de domínio sem viés de seleção, o artigo precisa de um teste ainda não usado (por exemplo, o conjunto de webcam previsto no ADR-0014). Todas as comparações usam uma seed só.

