# ADR-0006: Documentação e contexto do Claude versionados; uso restrito de subagentes

- **Status:** Proposta
- **Data:** 2026-10-22

## Contexto

O pesquisador quer acompanhar cada decisão, manter um relatório atualizado e versionar no GitHub o contexto usado pelo Claude Code e pelos agentes. A memória automática do Claude Code fica no perfil do usuário (`~/.claude/projects/...`), não no repositório.

## Alternativas consideradas

1. **Depender da memória automática do Claude:** não versionada, invisível para colegas e orientador.
2. **Contexto explícito no repositório:** `CLAUDE.md`, `docs/` e `.claude/agents|commands`. Fica auditável e legível por humanos.
3. **Muitos subagentes especializados** (programador, documentador, testador...) **vs. poucos subagentes de papel bem delimitado.** Cada subagente começa sem contexto e precisa redescobrir o que a sessão principal já sabe.

## Decisão

- Usar a alternativa 2. Todo conhecimento relevante vive em `CLAUDE.md` e `docs/`.
- Ter apenas dois subagentes:
  - `pesquisador-literatura`: buscas longas de artigos e datasets, executadas em isolamento para não poluir o contexto principal. Produz fichas em `docs/literatura/` e `docs/datasets/`.
  - `revisor-experimentos`: só leitura. Faz uma revisão crítica independente procurando vazamento de dados, divisões incorretas, números sem log e problemas de reprodutibilidade.
- A documentação é escrita pela sessão principal, que tem o contexto da decisão, por meio dos comandos `/registrar-decisao` e `/fechar-sessao`.

## Consequências

- Os arquivos `.claude/agents/*.md` e `.claude/commands/*.md` são revisados como código.
- Se surgir uma tarefa repetitiva e isolável (por exemplo, extração em lote), avaliar um novo subagente com um ADR próprio.
