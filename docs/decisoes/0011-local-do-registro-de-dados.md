# ADR-0011: Onde vive o registro de dados

- **Status:** Aceita (aprovada pelo pesquisador em 2026-10-24)
- **Data:** 2026-10-24
- **Relacionadas:** ADR-0005, ADR-0010, ADR-0013

## Contexto

O registro canônico (ADR-0010) mistura partes de naturezas diferentes:

- o esquema e os mapeamentos de rótulos são pequenos, não sensíveis e precisam de revisão versionada;
- as tabelas de amostras e sinalizadores são geradas pelos adaptadores a partir dos dados brutos. No futuro, a de sinalizadores terá dados pessoais de parceiros (LGPD).

A regra 6 do projeto diz que dados nunca entram no git. É preciso definir o que, do registro, conta como "dado".

## Alternativas consideradas

1. **Tudo em `LIBRAS_DATA_ROOT`:** cumpre a regra 6 sem discussão, mas o esquema e os mapeamentos de rótulos perdem histórico e revisão por *commit*.
2. **Tudo no git:** histórico completo, mas coloca dados pessoais de sinalizadores e conteúdo de terceiros de licença incerta no repositório.
3. **Divisão por natureza:** no git, o que é código ou curadoria própria e não sensível; em `LIBRAS_DATA_ROOT`, o que é gerado a partir dos dados ou é pessoal.

## Decisão

Alternativa 3:

| No git | Em `LIBRAS_DATA_ROOT/registry/` |
|---|---|
| Esquema e validação (código) | `samples`, `signers` (gerados pelos adaptadores) |
| `sources` | *Snapshots* versionados (`v0.1/`, ...) com hash |
| `label_map` e `glosses` dos datasets públicos | Vocabulário-alvo derivado do Signbank UFSC, até a licença ser esclarecida (ADR-0013) |
| | Qualquer dado pessoal de sinalizadores de parceiros |

## Justificativa

- O que vai para o git é curadoria feita pelo projeto (mapeamentos de rótulos) ou metadado público das fontes. Não são vídeos nem dados pessoais, e esses arquivos se beneficiam de revisão por *diff*.
- `samples` e `signers` são regeneráveis a partir dos dados brutos e dos adaptadores. Os *snapshots* com hash garantem a reprodutibilidade sem versioná-los no git.

## Consequências

- Esclarece a regra 6: "dados" inclui vídeos, landmarks, checkpoints, tabelas geradas e qualquer dado pessoal. Mapeamentos e esquema não são dados.
- Quando houver dados de parceiros, reavaliar DVC ou outro versionamento de dados (já previsto no ADR-0005).
- `LIBRAS_DATA_ROOT` nesta máquina: `/mnt/hd1tb/libras-data` (HD de 1 TB, adicionado em 2026-10-23).
