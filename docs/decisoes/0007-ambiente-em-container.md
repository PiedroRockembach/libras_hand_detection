# ADR-0007: Ambiente em container Docker (abordagem híbrida)

- **Status:** Proposta
- **Data:** 2026-10-22
- **Relacionadas:** ADR-0003, ADR-0005

## Contexto

O projeto roda em pelo menos duas máquinas: um notebook Windows 11 com RTX 4080 Laptop (desenvolvimento) e uma máquina Linux com RTX 5090 (treino). Talvez rode também em servidores do laboratório. O objetivo é ter um ambiente idêntico e independente de sistema operacional, que permita a terceiros reproduzir os resultados do artigo.

O `uv.lock` já fixa as dependências Python, mas não fixa o que está fora do Python: sistema, ffmpeg e bibliotecas nativas do OpenCV e do MediaPipe. A versão do ffmpeg pode mudar a decodificação dos quadros e, por consequência, os landmarks extraídos.

## Alternativas consideradas

1. **Ambiente nativo com `uv` em cada máquina:** simples, mas as dependências de sistema variam entre Windows e Linux.
2. **Todo o projeto em container, inclusive a demo:** máxima uniformidade. Porém repassar a webcam e as janelas da interface para o container no Windows (via `usbipd` e WSLg) é frágil, e no macOS não há GPU NVIDIA.
3. **Abordagem híbrida:** extração, treino e avaliação rodam em container (Dev Container no VS Code e `docker compose` para execuções sem interface); a demo de webcam roda de forma nativa. O `uv.lock` é a fonte única de dependências para os dois casos.

## Decisão

Alternativa 3:
- `docker/Dockerfile`: `python:3.12-slim-bookworm` + `uv` fixado + ffmpeg e bibliotecas nativas. A venv fica em `/opt/venv`, fora do workspace. O CUDA não é instalado na imagem, porque as wheels `cu128` trazem o runtime e o host só precisa do driver.
- `.devcontainer/devcontainer.json` para o desenvolvimento no VS Code (com GPU).
- `compose.yaml` para execuções sem interface na máquina Linux.
- Dados montados em `/data` a partir da variável `LIBRAS_DATA_ROOT` do host.
- No Windows, o repositório e os dados ficam **dentro do WSL2**, não em `C:\` nem no OneDrive.

## Justificativa

- Ambiente idêntico na 4080 (via WSL2) e na 5090 (Docker nativo no Linux), e portável para clusters (o Apptainer converte imagens Docker).
- A imagem ou o Dockerfile serve como artefato de reprodutibilidade do artigo.
- Manter o `uv.lock` como fonte única evita duas listas de dependências divergentes.
- A perda de desempenho de GPU em container é desprezível para treino.

## Consequências

- No Windows o fluxo muda: o repositório deve ser clonado no WSL2. A cópia no OneDrive pode ser abandonada.
- Pré-requisitos no host: Linux precisa de driver NVIDIA + NVIDIA Container Toolkit; Windows precisa de Docker Desktop com backend WSL2 + driver NVIDIA.
- macOS roda o container só em CPU (desenvolvimento, não treino). A imagem é `linux/amd64`.
- A demo em tempo real terá instruções de instalação nativa à parte (Fase 5).
- A justificativa do ADR-0003 foi revisada: com Linux no container, o TensorFlow teria GPU, mas a escolha do PyTorch se mantém pelos outros motivos.
