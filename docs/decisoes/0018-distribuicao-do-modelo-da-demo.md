# ADR-0018: Distribuição do modelo da demo por GitHub Release

- **Status:** Proposta
- **Data:** 2026-11-13
- **Relacionadas:** ADR-0005, ADR-0007, ADR-0016

## Contexto

A demo v0 (ADR-0016) precisa rodar em outras máquinas, a começar pelo notebook Windows do desenvolvimento, nativa e fora do container (ADR-0007). Um clone do repositório traz o código, mas não o modelo treinado: o checkpoint fica em `LIBRAS_DATA_ROOT/experiments/` (ADR-0005; regra 6 do CLAUDE.md). Os modelos do MediaPipe já são baixados pelo pacote (`python -m libras.extract models`).

O modelo da demo (`experiments/a6_gru_mirror_kinect/final-20261031T174312Z-770f082`) tem 4 arquivos, cerca de 5 MB: `config.yaml`, `meta.json`, `model.pt` e `history.json`. Ele foi treinado com o MINDS-Libras, que é CC BY 4.0 nos dois registros Zenodo ([ficha](../datasets/minds-libras.md)). Isso permite redistribuir um modelo derivado, com atribuição. O repositório é público.

## Alternativas consideradas

1. **Copiar a pasta à mão (pendrive, Drive, scp):** não muda nada no repositório. Por outro lado, não fica registrado qual modelo foi copiado, nada confere a integridade da cópia e cada máquina nova repete o processo.
2. **Commitar o checkpoint (ou um `.tar.gz`) no git:** é a opção mais simples para quem clona. Mas contraria o ADR-0005, e o binário fica no histórico para sempre: cada modelo novo da demo soma o seu tamanho a todo clone futuro, e removê-lo exige reescrever o histórico.
3. **Git LFS:** integrado ao `git clone`. A cota grátis de banda do GitHub é limitada, quem clona precisa ter o LFS instalado, e o checkpoint continua acoplado ao repositório de código.
4. **Hugging Face Hub:** é o padrão da área para publicar modelos, com *model card*. Exige mais uma conta e mais um serviço, o que é desproporcional para um modelo de demonstração com 20 sinais.
5. **Asset de uma GitHub Release, baixado por um script versionado:** fica fora do histórico do git, tem uma versão explícita (a tag) e cabe nos limites do GitHub para assets de release. Contras: o upload é manual (`gh release create`), e o script e a configuração precisam ser mantidos.

## Decisão

Alternativa 5.

- `configs/demo/<versão>.json` fixa a execução, os arquivos, o repositório, a tag, o nome do asset e os SHA-256 do arquivo e de cada arquivo. Para a demo v0, o arquivo é `configs/demo/v0.json`, com tag `demo-v0`.
- `scripts/demo_model.py pack` roda na máquina de treino. Ele gera um `.tar.gz` determinístico em `LIBRAS_DATA_ROOT/releases/`, grava os hashes na configuração e registra o evento em `PROVENANCE.jsonl`. O arquivo vai para a release com `gh release create`.
- `scripts/demo_model.py fetch` roda na máquina da demo, em Windows ou Linux. Ele baixa o asset, confere os hashes, escreve só os arquivos listados na configuração em `LIBRAS_DATA_ROOT/<run>` e registra o evento em `PROVENANCE.jsonl`.
- As notas da release levam a atribuição ao MINDS-Libras (CC BY 4.0) e citam o ADR-0016.

## Justificativa

- Mantém o ADR-0005: o git guarda só código e a configuração com os hashes. O binário fica num lugar com versão e integridade conferível.
- O script usa só a biblioteca padrão e `pathlib`, sem shell. Por isso roda no Windows sem nada além do `uv sync` que a demo já exige.
- O `.tar.gz` determinístico (datas e donos fixos nos cabeçalhos tar e gzip) faz com que os mesmos arquivos da execução gerem sempre o mesmo SHA-256. Assim, dá para conferir que o asset publicado corresponde à execução registrada.
- O `fetch` não usa caminhos vindos do arquivo, só os nomes da configuração. Com isso, um asset adulterado não escreve fora da pasta da execução.

## Consequências

- Rodar a demo numa máquina nova passa a ser: clonar, `uv sync`, `python -m libras.extract models`, `scripts/demo_model.py fetch`. O passo a passo está em [docs/demo.md](../demo.md).
- **Reprodução:** o `fetch` não entra em `scripts/prepare_data.sh`. Na reprodução a partir do zero, o modelo é retreinado com o config do experimento (ADR-0016). O download é só distribuição para máquinas de demonstração, e mesmo assim fica registrado em `PROVENANCE.jsonl`.
- **Cada modelo novo da demo** recebe uma nova `configs/demo/<versão>.json` e uma nova tag. Não se substitui o asset de uma tag existente, porque o hash da configuração deixaria de bater.
- **Ordem da publicação:** commit do script → `pack` (com a árvore limpa, para o log registrar o commit certo) → commit da configuração com os hashes → push → `gh release create`.
- Se o repositório deixar de ser público, o `fetch` anônimo para de funcionar e passa a exigir autenticação (`gh release download`). Nesse caso, este ADR deve ser revisto.
