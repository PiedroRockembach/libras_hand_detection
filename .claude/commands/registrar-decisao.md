---
description: Registra uma decisão de projeto como ADR em docs/decisoes/
argument-hint: <resumo da decisão>
---

Registre a seguinte decisão como ADR: $ARGUMENTS

1. Descubra o próximo número livre em `docs/decisoes/` e crie `NNNN-titulo-curto.md` a partir de `docs/decisoes/0000-template.md`.
2. Preencha contexto, alternativas consideradas (pelo menos duas, com prós e contras), decisão, justificativa e consequências, usando o que foi discutido nesta conversa. Não invente evidências. Se faltar informação, pergunte.
3. Status: `Proposta`, a menos que o pesquisador tenha aprovado explicitamente nesta conversa (nesse caso, `Aceita`, com a data).
4. Se a decisão substitui um ADR anterior, atualize o status do antigo para `Substituída por ADR-NNNN`.
5. Adicione a linha na tabela de decisões de `docs/README.md`.
6. Se a decisão afeta a metodologia, atualize o capítulo correspondente em `docs/relatorio/`.
