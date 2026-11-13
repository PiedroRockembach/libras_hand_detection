# Demo v0: reconhecimento de sinais isolados pela webcam

A demo reconhece os **20 sinais do MINDS-Libras** com o modelo `a6` (GRU com espelhamento e vista Kinect; relatório §4.2), treinado com os 12 sinalizadores (`experiments/a6_gru_mirror_kinect/final-20261031T174312Z-770f082`).

A estimativa de acerto com uma pessoa nova, no estúdio do MINDS, é a do LOSO: top-1 de 0,963 ± 0,025. Fora do estúdio (MALTA), o top-1 é de 0,729 ± 0,025. Na webcam, ainda não foi medido; para isso serve a gravação descrita abaixo.

## Como funciona

1. A câmera envia quadros. Para cada quadro, o MediaPipe extrai pose, mãos e rosto, com os mesmos recortes guiados pela pose da extração `mp_v2`.
2. **O corte dos sinais é automático:** um sinal começa quando um punho sobe acima da altura dos ombros mais 1,2 largura de ombros, e termina depois de cerca de 10 quadros com os braços abaixados. Como nos vídeos do MINDS, **comece e termine cada sinal com os braços em repouso**.
3. O trecho passa pelo mesmo pré-processamento do treino, e a tela mostra os 5 sinais mais prováveis.

Teclas: **espaço** marca o início e o fim de um sinal à mão; **a** liga e desliga o corte automático; **q** sai.

## Requisitos (rodar fora do container)

A demo precisa da webcam e de uma janela, então roda direto no host (ADR-0007). O modelo treinado não está no git: ele é baixado da GitHub Release `demo-v0` por `scripts/demo_model.py fetch`, que confere os SHA-256 fixados em `configs/demo/v0.json` (ADR-0018).

**Windows (PowerShell), numa máquina nova:**

```
winget install astral-sh.uv            # gerenciador de ambiente do projeto (reabra o terminal depois)
git clone -b develop https://github.com/PiedroRockembach/libras_hand_detection.git
cd libras_hand_detection
uv sync                                # cria .venv com Python 3.12 e as dependências do uv.lock
$env:LIBRAS_DATA_ROOT = "C:\libras-data"
uv run python -m libras.extract models --config configs/extraction/mp_v2.yaml
uv run python scripts/demo_model.py fetch --config configs/demo/v0.json
uv run python -m libras.demo live --run experiments/a6_gru_mirror_kinect/final-20261031T174312Z-770f082
```

`$env:` vale só para a sessão do terminal. Para fixar a variável: `setx LIBRAS_DATA_ROOT C:\libras-data`.

**Linux:**

```
sudo pacman -S uv                      # ou: curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/Documentos/projetos/libras_hand_detection
uv sync
export LIBRAS_DATA_ROOT=/mnt/hd1tb/libras-data
uv run python -m libras.extract models --config configs/extraction/mp_v2.yaml   # se ainda não tiver
uv run python scripts/demo_model.py fetch --config configs/demo/v0.json         # se ainda não tiver
uv run python -m libras.demo live --run experiments/a6_gru_mirror_kinect/final-20261031T174312Z-770f082
```

- No Linux, a webcam precisa aparecer como `/dev/video0`. Se houver mais de uma câmera, use `--camera 1` (vale também no Windows).
- Os modelos do MediaPipe são lidos de `LIBRAS_DATA_ROOT/models/mediapipe/`, os mesmos da extração.
- O `fetch` e o `extract models` não repetem o download quando os arquivos já estão lá, e os dois registram o evento em `PROVENANCE.jsonl`.
- A inferência roda na CPU (`--device cpu`, o padrão), o que basta para o modelo de landmarks.

Para testar sem webcam, em qualquer ambiente, inclusive no container, há o modo de vídeo:

```
python -m libras.demo video ARQUIVO.mp4 --run ...            # corte automático
python -m libras.demo video ARQUIVO.mp4 --run ... --whole    # o vídeo inteiro é um sinal
```

## Gravação do conjunto de teste de webcam (`webcam_v0`)

O MALTA foi usado para escolher a configuração do modelo, então deixou de ser um teste neutro (relatório §4.2). As gravações de webcam servem de **teste fora de domínio nunca usado na escolha**.

```
uv run python -m libras.demo record --run ... --signer p01 --takes 3 --consent-confirmed
```

- A tela mostra cada um dos 20 sinais. Comandos: **espaço** começa e termina a gravação; **r** refaz; **s** pula o sinal; **q** sai.
- Os vídeos vão para `LIBRAS_DATA_ROOT/raw/webcam_v0/<sinalizador>/`, com o índice `takes.csv`. Cada sessão é registrada em `PROVENANCE.jsonl`.
- **Use um identificador pseudônimo** (`p01`, `p02`...) e só grave quem concordou. A flag `--consent-confirmed` é obrigatória. O termo de consentimento ainda precisa ser definido com o laboratório (a verificar).
- Quem grava precisa saber os sinais: uma pessoa fluente, ou alguém treinado com os vídeos do MINDS. Anote no diário quem foi.
- **Pendente:** um adaptador `webcam_v0` no registro (ADR-0010) e a avaliação dos modelos do LOSO nesses vídeos.
