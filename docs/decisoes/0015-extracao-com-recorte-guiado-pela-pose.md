# ADR-0015: Extração de landmarks com recorte guiado pela pose (`mp_v2`)

- **Status:** Aceita (aprovada pelo pesquisador em 2026-10-29)
- **Data:** 2026-10-28
- **Relacionadas:** ADR-0002, ADR-0012, ADR-0014

## Contexto

Piloto de extração do ADR-0012, rodado na imagem oficial `libras:dev` (MediaPipe 1.1.1, OpenCV 5.0.0). O conjunto foram 424 vídeos:

- sinalizador 5 do MINDS, Canon e Kinect (100 + 100);
- os 124 clipes do MALTA com os 20 sinais do MINDS;
- 100 vídeos do informante 1 do EQT.

Logs: `landmarks/mp_v1/stats.csv`, `landmarks/mp_v2/stats.csv` e `PROVENANCE.jsonl`.

A métrica principal é a **mão perdida**: em quadros onde o modelo de pose vê um pulso dentro da imagem, a fração em que nenhuma mão detectada está perto dele. Ela foi separada em **sinalizando** (pulso acima do ponto médio entre ombros e quadril) e **repouso** (braço abaixado).

**`mp_v1`, quadro inteiro, mediana por vídeo:**

| Fonte | Mão perdida | Rosto detectado | ms/quadro |
|---|---|---|---|
| MINDS Canon | 36% | **0%** | 65 |
| MINDS Kinect | 6% | **0%** | 66 |
| EQT | 17% | 14% | 111 |
| MALTA (224 px) | 37% | 100% | 62 |

O detector de rosto falha quando a pessoa está longe da câmera num quadro de alta resolução.

## Alternativas consideradas

1. **`mp_v1`, quadro inteiro:** mais rápida, mas sem rosto no MINDS e no EQT, e o rosto carrega expressões não-manuais (ADR-0002).
2. **`mp_v2`, recorte guiado pela pose:** a pose roda no quadro inteiro. Mãos e rosto rodam em recortes fixos por amostra (espaço de sinalização e cabeça), derivados da pose, e as coordenadas voltam ao quadro inteiro.
3. **`HolisticLandmarker` da API Tasks:** existe nesta versão do MediaPipe (verificado em 2026-10-28), mas não foi testado. Fica como comparação futura.
4. **Modelo de pose mais pesado (`heavy`):** a pose já é detectada em ~100% dos quadros. Não ataca o problema.

## Decisão

**`mp_v2`** (`configs/extraction/mp_v2.yaml`) é a configuração padrão de extração.

**Resultados do piloto, `mp_v1` → `mp_v2`, nos mesmos 424 vídeos:**

| Fonte | Rosto | Mão perdida **sinalizando** | Mão perdida em repouso | ms/quadro |
|---|---|---|---|---|
| MINDS Canon | 0% → **100%** | 7,8% → **6,6%** | 75,7% → 75,5% | 65 → 90 |
| MINDS Kinect | 0% → **100%** | 22,1% → **16,3%** | 4,5% → 4,0% | 66 → 91 |
| EQT | 14% → **100%** | 6,9% → **6,4%** | 33,9% → 35,6% | 111 → 184 |
| MALTA | 100% → 100% | 17,1% → **18,0%** | 64,3% → 58,8% | 62 → 63 |

As taxas de mão perdida somam todos os quadros com pulso visível de cada fonte. As colunas de rosto e ms/quadro são medianas por vídeo. Para reproduzir: `python -m libras.extract report --config configs/extraction/mp_v{1,2}.yaml`.

## Justificativa

- O recorte resolve o rosto por completo. Também reduz a perda de mãos durante o sinal no MINDS (Canon e, sobretudo, Kinect) e no EQT, com uma piora pequena no MALTA (+0,9 ponto).
- A perda de mãos medida no total vinha quase toda da posição de repouso. **Durante o sinal** a perda é baixa no MINDS Canon e no EQT (~6%) e moderada no MALTA (18%).
- O custo extra (~40% no MINDS, ~65% no EQT) é pago uma única vez (ADR-0012).
- A extração é determinística: reextrair um vídeo da `mp_v1` depois da refatoração do código deu resultado idêntico bit a bit.

## Consequências

- **Vista principal do MINDS: Canon.** O Kinect perde mais mãos durante o sinal (16,3%, contra 6,6%), possivelmente por borrão de movimento (a verificar). Ele entra como vista extra para aumento de dados (ADR-0014).
- **No carregamento** (ADR-0012/0014): interpolar falhas curtas de mão (limite a definir); medir a qualidade da amostra pela perda **durante o sinal**, não pelo total; considerar excluir ou reduzir o peso de amostras acima de um limiar.
- **MALTA:** a perda de 18% durante o sinal, variando por subcoleção, limita sua precisão como teste fora de domínio. Os resultados no MALTA devem ser reportados por subcoleção.
- **Extração completa estimada** (8 processos, ms/quadro do piloto): MINDS Canon ~30 min, Kinect ~35 min, EQT ~3,5 h. O MALTA só precisa dos clipes com rótulo mapeado.
- **Em aberto:** comparar com o `HolisticLandmarker`; resolver a lateralidade das mãos quando o MediaPipe atribui o mesmo rótulo às duas; testar a perda durante o sinal com limiares de confiança mais baixos.
