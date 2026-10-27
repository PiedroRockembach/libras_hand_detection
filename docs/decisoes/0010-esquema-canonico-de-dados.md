# ADR-0010: Esquema canônico de dados com adaptadores por fonte

- **Status:** Aceita (aprovada pelo pesquisador em 2026-10-24)
- **Data:** 2026-10-24
- **Relacionadas:** ADR-0004, ADR-0009, ADR-0011, ADR-0012, ADR-0013

## Contexto

Cada dataset tem sua estrutura de pastas, nomes de rótulos, identificação de sinalizadores e metadados. O ADR-0009 pede uma base que cresça com fontes novas (datasets públicos agora, gravações de parceiros no futuro) e que permita medir lacunas de vocabulário e de diversidade. Também é preciso registrar licença e consentimento por amostra, para separar o que pode ser usado em treino, em demo, em redistribuição ou em uso comercial.

## Alternativas consideradas

1. **Cada dataset no formato original, com um *loader* por dataset:** sem trabalho inicial, mas cada fonte nova multiplica o código. Não permite consultas transversais (cobertura por sinal, por sinalizador, por condição de gravação).
2. **Converter tudo para um formato único, copiando e recodificando os vídeos:** uniforme, mas duplica centenas de GB e perde a correspondência direta com o original.
3. **Registro canônico de metadados que aponta para os arquivos originais, preenchido por um adaptador por fonte:** os vídeos ficam onde foram baixados, e só os metadados são padronizados.

## Decisão

Alternativa 3. O registro tem cinco tabelas:

| Tabela | Uma linha por | Campos principais |
|---|---|---|
| `sources` | fonte (dataset ou parceiro) | `source_id`, nome, versão, URL/DOI, licença, `redistribution` (bool), `commercial_use` (bool), ficha em `docs/datasets/` |
| `glosses` | sinal do vocabulário canônico | `gloss_id` estável, glosa normalizada, variante, `signbank_ref`, `ines_ref`, observações |
| `label_map` | rótulo de uma fonte | `source_id`, rótulo original → `gloss_id` |
| `signers` | pessoa em uma fonte | `signer_uid` global, `source_id`, id original, `same_person_as` (ligação entre fontes), perfil surdo/ouvinte, região, faixa etária, mão dominante (vazio quando a fonte não informa) |
| `samples` | clipe de um sinal | `sample_id`, `source_id`, `gloss_id`, `signer_uid`, caminho relativo a `data_root`, quadros de início e fim (para vídeos com vários sinais), fps, resolução, nº de quadros, câmera, fundo, iluminação, licença, escopo do consentimento, hash SHA-256 do vídeo, *flags* de qualidade |

- **Adaptadores:** um módulo por fonte (`minds`, `eqt_uece`, `malta`, ...) que lê o formato original e gera linhas nas tabelas. Para parceiros, haverá um **formato de submissão** padrão (pastas + planilha nos mesmos campos + termo de consentimento), definido no mês 4 (ADR-0009).
- **Divisões:** o grupo de divisão é sempre `signer_uid`, resolvido via `same_person_as`, para que a mesma pessoa não apareça no treino e no teste mesmo vindo de fontes diferentes (ADR-0004).
- **Versões:** o registro é publicado em *snapshots* numerados (v0.1, v0.2, ...) com hash. Todo experimento registra o *snapshot* usado.
- **Formato de arquivo:** CSV UTF-8, validado por esquema em código. É legível em planilha, o que facilita a troca com parceiros. Migrar para Parquet se o tamanho exigir.

## Justificativa

- Os adaptadores isolam a parte que muda (o formato de cada fonte) da parte que fica (treino, avaliação, relatórios).
- Licença e consentimento por amostra permitem montar subconjuntos por finalidade com um filtro, em vez de decidir dataset a dataset.
- O hash do vídeo detecta duplicatas exatas entre fontes. O MALTA-LIBRAS agrega fontes públicas e pode conter vídeos já presentes em outros datasets (a verificar).

## Consequências

- Primeiro código do pacote: esquema, validação e adaptadores do MINDS e do EQT-UECE, mais um inventário que preenche os "a verificar" das fichas (nº efetivo de amostras, fps, resolução).
- A ligação de sinalizadores entre fontes (`same_person_as`) é manual no início. Detecção de duplicatas visualmente parecidas (hash perceptual) fica para depois.
- Metadados de sinalizadores de parceiros são dados pessoais (LGPD): ver ADR-0011.
- **Atualização (2026-10-24, implementação):** o inventário levou a duas colunas a mais. `glosses.unit` (`sign` | `phrase`) existe porque o LIBRAS-EQT-UECE mistura sinais isolados e frases inteiras. `samples.media` (`video` | `frames_tensor`) existe porque o MALTA-LIBRAS não distribui vídeo, só tensores de 32 quadros 224×224. O MINDS-Libras entra como duas fontes (`minds`, câmera Canon; `minds_kinect`, RGB do Kinect), com os sinalizadores ligados por `same_person_as`. As fontes sem identificação de sinalizador (MALTA) usam um sinalizador "desconhecido" por subcoleção.
