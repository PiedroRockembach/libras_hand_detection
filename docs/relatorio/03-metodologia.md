# 3. Metodologia

## 3.1 Visão geral

```
vídeo → landmarks (MediaPipe Tasks: mãos, pose, face parcial) → normalização → modelo temporal → sinal
```

Sinais estáticos e dinâmicos são tratados de forma unificada, como sequências de landmarks (ADR-0002).

## 3.2 Dados

*A definir na Fase 1 (ver [docs/datasets/](../datasets/)).*

## 3.3 Extração e pré-processamento

*Fase 2.*

## 3.4 Modelos

*Fases 3–4.* Progressão planejada: MLP sobre estatísticas da sequência → GRU bidirecional → transformer / grafo espaço-temporal.

## 3.5 Protocolo de avaliação

Independente de sinalizador (ADR-0004): protocolo oficial do dataset quando houver; caso contrário, *leave-one-signer-out*. Métricas: acurácia top-1/top-5, F1-macro, matriz de confusão.

## 3.6 Ambiente e reprodutibilidade

- PyTorch ≥ 2.7 com CUDA 12.8 (ADR-0003). Treino em RTX 5090 (Linux); desenvolvimento em RTX 4080 Laptop (Windows/WSL2).
- Ambiente em container Docker (Debian bookworm, Python 3.12, ffmpeg do sistema), com dependências fixadas em `uv.lock` (ADR-0007).
- Configurações em YAML, seed fixa e hash do commit registrados por experimento.
