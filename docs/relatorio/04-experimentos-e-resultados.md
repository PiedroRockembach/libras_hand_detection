# 4. Experimentos e resultados

Cada experimento entra nesta tabela com link para a configuração e o identificador da execução. Números só entram aqui se vierem de um log de experimento (`LIBRAS_DATA_ROOT/experiments/<nome>/<execução>/summary.json`).

| ID | Data | Dataset / divisão | Modelo | Entrada | Top-1 | F1-macro | Config | Commit |
|---|---|---|---|---|---|---|---|---|
| `a0_stats_mlp_minds/20261030T175856Z-1b97ce9` | 2026-10-30 | MINDS Canon, LOSO 12 dobras | MLP sobre estatísticas temporais (531 mil parâmetros) | landmarks `mp_v2`, 64 quadros × 446 | 0,933 ± 0,052 | 0,920 ± 0,066 | [a0](../../configs/experiments/a0_stats_mlp_minds.yaml) | `1b97ce9` |
| `a1_gru_minds/20261030T181123Z-1b97ce9` | 2026-10-30 | MINDS Canon, LOSO 12 dobras | GRU bidirecional 2 × 192 + cosseno (1,30 M parâmetros) | idem | 0,934 ± 0,044 | 0,919 ± 0,054 | [a1](../../configs/experiments/a1_gru_minds.yaml) | `1b97ce9` |

Média ± desvio-padrão (populacional) entre as 12 dobras. Em cada dobra, o teste é um sinalizador e a época é escolhida pela acurácia num outro sinalizador de validação (ADR-0004). O F1-macro de cada dobra considera as classes presentes no teste ou previstas, porque os sinalizadores 03, 04 e 09 não têm alguns sinais.

## 4.1 Fase A: baseline de sinais isolados (MINDS)

**Teste fora de domínio (MALTA, clipes com os 20 sinais do MINDS), modelos das 12 dobras:**

| Modelo | Top-1 | Top-5 | F1-macro |
|---|---|---|---|
| a0 (MLP estatísticas) | 0,545 ± 0,042 | 0,794 ± 0,012 | 0,510 ± 0,052 |
| a1 (GRU) | 0,658 ± 0,036 | 0,812 ± 0,025 | 0,629 ± 0,045 |

Top-1 médio por subcoleção do MALTA (a0 → a1): acessibilidade_2 (20 clipes) 0,671 → 0,675; acessibilidade_3 (13) 0,500 → 0,603; spread_the_sign (14) 0,482 → 0,565; ufsc (19) 0,487 → 0,596; ufsc_v2 (18) 0,426 → 0,620; ufv (6) 0,222 → 0,444; usp (8) 0,625 → 0,792; youtube (13) 0,679 → 0,801; youtube_v2 (13) 0,679 → 0,801.

**Leitura:**

- No MINDS, o modelo que ignora a ordem dos quadros (a0) empata com a GRU (a1). Os 20 sinais do MINDS se separam pelas posições médias. A acurácia está na faixa relatada por dos Santos 2025 (0,94 ± 0,04, MediaPipe), mas o protocolo dele é LOPO aninhado, então a comparação não é direta.
- **Fora de domínio, a GRU é melhor (0,658 contra 0,545).** A queda de ~0,93 para ~0,66 confirma que a acurácia no MINDS não mede robustez (Delucis 2025), e é este o número a melhorar antes da demo.
- **Reprodutibilidade:** as execuções a partir do commit limpo repetiram exatamente os números de uma execução anterior com o mesmo código ainda não commitado (determinismo em GPU com seed fixa).

**Limitação do conjunto MALTA usado:** os 124 clipes são **109 clipes distintos**. A subcoleção `youtube_v2` é uma cópia da `youtube`: 1.183 dos 1.185 arquivos distintos de `youtube_v2` são idênticos byte a byte a arquivos de `youtube`. A própria `youtube` também tem nomes duplicados por acentuação corrompida. Os resultados acima contam essas cópias. A correção (remover as duplicatas na seleção do MALTA) e a reavaliação estão pendentes. Se `ufsc` e `ufsc_v2` também se sobrepõem, ainda está a verificar.
