# 3. Metodologia

## 3.1 Visão geral

```
vídeo → landmarks (MediaPipe Tasks: mãos, pose, face parcial) → normalização → modelo temporal → sinal
```

Sinais estáticos e dinâmicos são tratados de forma unificada, como sequências de landmarks (ADR-0002).

## 3.2 Dados

*ADR-0009 (Aceita) e ADR-0008 (Proposta).* Datasets de Libras de acesso aberto baixados: MINDS-Libras (principal), LIBRAS-EQT-UECE, MALTA-LIBRAS (teste fora de domínio) e LSWH100 (configurações de mão). Ver [docs/datasets/](../datasets/).

Todas as fontes são integradas por um **registro canônico** (ADR-0010): tabelas de fontes, vocabulário de glosas, mapeamento de rótulos, sinalizadores e amostras, preenchidas por um adaptador por fonte. Licença e consentimento são registrados por amostra. A divisão treino/teste agrupa por sinalizador mesmo entre fontes diferentes. Um **vocabulário-alvo** baseado no Signbank da Libras (ADR-0013) permite medir as lacunas de vocabulário e de diversidade.

### 3.2.1 Inventário e viabilidade (2026-10-27)

Números obtidos com `python -m libras.registry build` sobre os dados baixados. O registro tem 7.705 amostras de 38 sinalizadores. O MALTA-LIBRAS ainda não entra porque os rótulos não foram mapeados.

| Fonte | Amostras | Sinalizadores | Vídeo | Observações |
|---|---|---|---|---|
| MINDS-Libras, Canon | 1.158 (1.200 nominais) | 12 | 1920×1080, ~30,2 fps, 71–233 quadros | 20 sinais; 5 a 7 repetições por sinal e sinalizador; chroma key |
| MINDS-Libras, Kinect v2 | 1.200 | 12 (os mesmos) | 1920×1080, 30 fps, 150 quadros fixos | Aparentemente a mesma execução da Canon, de outro ângulo (a verificar) |
| LIBRAS-EQT-UECE v4 | 5.347 | 5 | 1920×1080, 30 fps, 33–279 quadros | 175 glosas (44 sinais isolados, 131 frases); fundo preto; 84 amostras com `label_conflict` |
| MALTA-LIBRAS (complete) | 25.653 clipes | não identificados | Tensores 32 × 224 × 224 | 9 subcoleções com pessoas, fundos e enquadramentos variados; ~13,4 mil rótulos |

Sobreposição de vocabulário, com glosas comparadas sem acentos nem pontuação:

- **MINDS × MALTA:** os 20 sinais do MINDS aparecem no MALTA, com 3 a 11 clipes cada (~120 no total), em 3 a 9 subcoleções.
- **EQT-UECE × MALTA:** 36 das 175 glosas.
- **MINDS × EQT-UECE:** nenhuma.

Avaliação de viabilidade por objetivo do ADR-0009:

- **Baseline com LOSO no MINDS:** viável. As classes são equilibradas e os rótulos estão limpos.
- **Demo de webcam com os 20 sinais do MINDS:** viável, mas com risco de queda de desempenho fora do estúdio. O MALTA permite medir essa queda nos mesmos 20 sinais, embora com poucos clipes por sinal (intervalos de confiança largos).
- **Treino conjunto MINDS + EQT-UECE:** sem glosas em comum, cada classe vem de um único dataset. Há risco de o modelo aprender um atalho por dataset (enquadramento, escala), e a normalização (ADR-0012) precisa neutralizá-lo. A avaliação no MALTA verifica se funcionou.
- **Generalização entre datasets:** viável nos pares MINDS → MALTA e EQT-UECE → MALTA.
- **Restrições de licença:** o MINDS (CC BY 4.0) pode ir para a demo. O EQT-UECE (CC BY-NC 4.0) e o MALTA (licença das fontes a verificar) exigem verificação antes de qualquer uso fora da pesquisa.

**Ainda em aberto:** a taxa de detecção de mãos do MediaPipe em cada fonte, principalmente no MINDS, em que a pessoa ocupa pouco do quadro, e no MALTA, em 224 px. Essa taxa decide a viabilidade de fato e será medida no piloto de extração.

## 3.3 Extração e pré-processamento

*Fase 2.* O cache guarda os landmarks brutos do MediaPipe (todos os pontos, fps original, máscara de detecção), um arquivo por amostra e versionado pelo extrator. Seleção de pontos, normalização, reamostragem e aumento de dados são aplicados no carregamento, configurados em YAML (ADR-0012).

## 3.4 Modelos

*Fases 3–4.* Progressão planejada: MLP sobre estatísticas da sequência → GRU bidirecional → transformer / grafo espaço-temporal.

## 3.5 Protocolo de avaliação

Independente de sinalizador (ADR-0004): protocolo oficial do dataset quando houver; caso contrário, *leave-one-signer-out*. Métricas: acurácia top-1/top-5, F1-macro, matriz de confusão.

## 3.6 Ambiente e reprodutibilidade

- PyTorch ≥ 2.7 com CUDA 12.8 (ADR-0003). Treino em RTX 5090 (Linux); desenvolvimento em RTX 4080 Laptop (Windows/WSL2).
- Ambiente em container Docker (Debian bookworm, Python 3.12, ffmpeg do sistema), com dependências fixadas em `uv.lock` (ADR-0007).
- Configurações em YAML, seed fixa e hash do commit registrados por experimento.
