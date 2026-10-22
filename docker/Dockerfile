# Ambiente oficial de extração, treino e avaliação (ADR-0007).
# Não precisa de CUDA na imagem: as wheels do PyTorch cu128 trazem o runtime;
# o host só precisa do driver NVIDIA (>= suporte a CUDA 12.8) e do container toolkit.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /uvx /bin/

# ffmpeg fixado pela imagem: a decodificação de vídeo afeta os landmarks extraídos.
# libgl1/libglib2.0-0/libegl1/libgles2: dependências de OpenCV e MediaPipe.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libegl1 libgles2 git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
    && mkdir -p /opt/venv /data /workspace \
    && chown -R ${USERNAME}:${USERNAME} /opt/venv /data /workspace

# venv fora do workspace montado, para não colidir com um .venv nativo do host.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:${PATH} \
    LIBRAS_DATA_ROOT=/data

USER ${USERNAME}
WORKDIR /workspace

# Dependências primeiro (camada em cache enquanto uv.lock não mudar).
# O cache do uv fica num cache mount do BuildKit: fora da imagem (~8 GB a menos)
# e reaproveitado entre builds.
COPY --chown=${USERNAME}:${USERNAME} pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/home/${USERNAME}/.cache/uv,uid=${USER_UID},gid=${USER_GID} \
    uv sync --frozen --no-install-project

COPY --chown=${USERNAME}:${USERNAME} . .
RUN --mount=type=cache,target=/home/${USERNAME}/.cache/uv,uid=${USER_UID},gid=${USER_GID} \
    uv sync --frozen
