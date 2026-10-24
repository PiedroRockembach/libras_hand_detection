# ADR-0005: Dados fora do git e fora do OneDrive

- **Status:** Proposta
- **Data:** 2026-10-22

## Contexto

O repositório fica numa pasta sincronizada pelo OneDrive. Os datasets de vídeo ocupam dezenas de GB, e o cache de landmarks e os checkpoints geram milhares de arquivos. Muitos datasets também têm licenças que proíbem redistribuição.

## Alternativas consideradas

1. **Dados dentro do repositório (git ou Git LFS):** esbarra em licença e cota, e o OneDrive sincronizaria tudo.
2. **DVC com armazenamento remoto:** versiona os dados, mas adiciona infraestrutura. Fica para quando houver dados próprios.
3. **Pasta local fora do OneDrive, indicada por configuração, mais scripts de download e extração documentados:** simples e reprodutível a partir das fichas em `docs/datasets/`.

## Decisão

Alternativa 3. O caminho vem de `configs/paths.yaml` (`data_root`, por exemplo `C:/dev/libras-data`), que não é versionado. Cada dataset tem uma ficha com origem, licença e procedimento de obtenção. Reavaliar o DVC quando houver coleta própria.

## Consequências

- Recomenda-se também criar o ambiente virtual fora do OneDrive, ou excluir `.venv/` da sincronização.
- **Atualização (ADR-0007):** dentro do container, os dados ficam em `/data`, montado a partir de `LIBRAS_DATA_ROOT` do host. No Windows, essa pasta e o repositório ficam no sistema de arquivos do WSL2.
- Para reproduzir, é preciso baixar os datasets manualmente seguindo as fichas.
