---
description: Encerra a sessão de trabalho atualizando diário, relatório e índice de decisões
---

Encerre a sessão de trabalho atual:

1. Crie ou atualize `docs/diario/AAAA-MM-DD.md` (data de hoje) com as seções **Feito**, **Observações** (problemas, surpresas, resultados parciais) e **Pendências / próximos passos** (checklist). Seja factual e cite arquivos, commits e IDs de experimentos.
2. Verifique se alguma decisão tomada nesta sessão ainda não tem ADR. Se houver, liste-as e pergunte se devem ser registradas (`/registrar-decisao`).
3. Atualize os capítulos de `docs/relatorio/` afetados pelo trabalho da sessão. Resultados só entram se houver log de experimento correspondente.
4. Se algo em `CLAUDE.md` ficou desatualizado (estrutura, regras, hardware), atualize.
5. Mostre um resumo do que foi alterado e sugira uma mensagem de commit. Não faça o commit sem pedir.
