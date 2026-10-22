# CLAUDE.md

Contexto para o Claude Code (e para qualquer pessoa nova no projeto). Mantenha curto e atualizado.

## O projeto

Pesquisa científica (laboratório universitário, com intenção de publicação) sobre **reconhecimento automático de sinais da Libras** a partir de vídeo, incluindo sinais **estáticos e dinâmicos** (com movimento). Meta de longo prazo: escalar para o vocabulário completo da Libras; o escopo inicial é reduzido (ver [ADR-0001](docs/decisoes/0001-escopo-e-plano-inicial.md)).

Abordagem atual: vídeo → landmarks MediaPipe (mãos + pose + face parcial) → modelo temporal em PyTorch. Tudo que for decidido está em [docs/decisoes/](docs/decisoes/).

## Regras de trabalho

1. **Toda decisão de projeto vira um ADR** em `docs/decisoes/` (use `/registrar-decisao`). Decisão = escolha entre alternativas que afeta resultados, método ou estrutura. Inclua as alternativas descartadas e o porquê.
2. **Status dos ADRs:** `Proposta` até o pesquisador (Piedro) aprovar explicitamente; só então `Aceita`. Não marque `Aceita` por conta própria.
3. **Ao fim de cada sessão de trabalho**, atualize o diário (`docs/diario/AAAA-MM-DD.md`) e os capítulos afetados de `docs/relatorio/` (use `/fechar-sessao`).
4. **Reprodutibilidade:** todo experimento roda a partir de um arquivo em `configs/`, com seed fixa, e registra o hash do commit. Nunca reporte um número que não esteja num log de experimento.
5. **Avaliação independente de sinalizador** (pessoas do teste nunca aparecem no treino). Qualquer exceção precisa de ADR.
6. **Dados nunca entram no git.** Vídeos, landmarks e checkpoints ficam em `LIBRAS_DATA_ROOT` (fora do OneDrive). Cada dataset tem uma ficha em `docs/datasets/`.
7. Não invente números, citações ou características de datasets. Se não foi verificado, escreva "a verificar".

## Idioma e convenções

- Documentação em **português (pt-BR)**; código, nomes de variáveis e docstrings em **inglês**.
- Python 3.12, pacote em `src/libras/`, dependências em `uv.lock` (fonte única).
- **Ambiente oficial = container** (ADR-0007): Dev Container no VS Code ou `docker compose run --rm libras ...`. Dentro dele: venv em `/opt/venv`, dados em `/data`. A demo de webcam é a exceção e roda nativa.
- Não adicione dependências com `pip install` no container: use `uv add` e faça commit do `uv.lock`.
- Commits pequenos e descritivos; branch de trabalho `develop`, `main` recebe marcos estáveis.

## Mapa do repositório

| Caminho | Conteúdo |
|---|---|
| `src/libras/` | Código do pacote (extração, features, modelos, treino, avaliação, demo) |
| `configs/` | Configurações YAML de extração e experimentos |
| `docker/`, `.devcontainer/`, `compose.yaml` | Ambiente em container |
| `docs/relatorio/` | Relatório vivo, um arquivo por capítulo |
| `docs/decisoes/` | ADRs numerados |
| `docs/diario/` | Diário de bordo por sessão |
| `docs/datasets/` | Fichas de datasets (origem, licença, estatísticas, download) |
| `docs/literatura/` | Fichamentos de artigos |
| `.claude/agents/` | Subagentes (`pesquisador-literatura`, `revisor-experimentos`) |
| `.claude/commands/` | Comandos `/registrar-decisao`, `/fechar-sessao` |

## Hardware

- Desenvolvimento: notebook Windows 11 com RTX 4080 Laptop (12 GB), Docker Desktop + WSL2.
- Treino: máquina Linux com RTX 5090 (Blackwell, sm_120) → exige PyTorch com CUDA ≥ 12.8 e driver compatível no host.

## Histórico

O código anterior (fork de Kazuhito00/hand-gesture-recognition-using-mediapipe e o protótipo do alfabeto de 2024) foi removido da árvore e preservado nas tags `v0-fork-original` e `v0-prototipo-alfabeto`. Ver [ADR-0001](docs/decisoes/0001-escopo-e-plano-inicial.md).
