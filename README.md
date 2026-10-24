# Reconhecimento de Sinais da Libras

Pesquisa sobre reconhecimento automático de sinais da Língua Brasileira de Sinais (Libras) a partir de vídeo, cobrindo sinais estáticos e dinâmicos.

> Projeto em reformulação (out/2026). O estado atual, as decisões e o andamento estão documentados em [docs/](docs/README.md).

## Abordagem

```
vídeo → landmarks (MediaPipe: mãos + pose + face) → normalização → modelo temporal (PyTorch) → sinal
```

## Estrutura

```
src/libras/     código do pacote
configs/        configurações de extração e experimentos
docs/           relatório, decisões (ADRs), diário, fichas de datasets e literatura
```

## Ambiente

O ambiente oficial é um container Docker com GPU ([ADR-0007](docs/decisoes/0007-ambiente-em-container.md)).

**Pré-requisitos:** driver NVIDIA com suporte a CUDA ≥ 12.8. No Linux, também Docker + NVIDIA Container Toolkit; no Windows, Docker Desktop com backend WSL2.
No Windows, clone o repositório **dentro do WSL2** (não em `C:\` nem no OneDrive).

```bash
export LIBRAS_DATA_ROOT=~/libras-data      # pasta de dados no host, montada em /data
cp configs/paths.example.yaml configs/paths.yaml
```

- **VS Code:** "Dev Containers: Reopen in Container".
- **Sem interface (ex.: treino via SSH):**
  ```bash
  docker compose run --rm libras python -c "import torch; print(torch.cuda.get_device_name())"
  ```
- **Nativo (só para a demo de webcam):** `uv sync`.

## Histórico

Versões anteriores (fork de [Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe), Apache-2.0, e o protótipo do alfabeto) estão nas tags `v0-fork-original` e `v0-prototipo-alfabeto`.
