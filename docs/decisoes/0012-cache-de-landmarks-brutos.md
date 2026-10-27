# ADR-0012: Cache de landmarks brutos, com normalização no carregamento

- **Status:** Aceita (aprovada pelo pesquisador em 2026-10-24)
- **Data:** 2026-10-24
- **Relacionadas:** ADR-0002, ADR-0009, ADR-0010

## Contexto

A extração com MediaPipe é a etapa mais cara do pré-processamento e deve rodar uma única vez por vídeo. O ADR-0002 prevê ablações de partes do corpo e de esquemas de normalização, e o ADR-0009 prevê fontes com fps, resolução e enquadramento diferentes. É preciso decidir o que fica gravado no cache e o que é calculado depois.

## Alternativas consideradas

1. **Gravar landmarks já normalizados e reamostrados:** carregamento mais simples, mas cada mudança de normalização, de fps ou de pontos selecionados exige reextrair tudo.
2. **Gravar landmarks brutos e aplicar toda a padronização no carregamento, configurada em YAML:** extração única. Ablações viram troca de configuração, ao custo de um pouco de processamento por época.
3. **Formato do arquivo:** um `.npz` por amostra, ou um arquivo grande por dataset (HDF5/Zarr).

## Decisão

Alternativa 2, com um `.npz` por amostra:

- **Conteúdo:** todos os pontos que o extrator fornece, sem selecionar (mãos 2 × 21, pose 33, face completa), em coordenadas de imagem normalizadas, com presença/visibilidade e lateralidade das mãos por quadro, mais uma máscara de detecção. Quadros no fps original, sem reamostragem.
- **Caminho:** `landmarks/<versão do extrator>/<source_id>/<sample_id>.npz`, com um manifesto por versão (versão do MediaPipe, hash dos modelos `.task`, parâmetros).
- **No carregamento (configurado em YAML):** seleção de pontos, normalização (centro e escala pelo corpo), reamostragem de fps, tratamento de quadros sem detecção e aumento de dados.

## Justificativa

- Reextrair centenas de GB de vídeo para testar uma normalização inviabilizaria as ablações.
- Gravar a face completa custa espaço, mas o subconjunto ideal de pontos da face ainda é uma questão aberta (ADR-0002).
- Um arquivo por amostra simplifica a extração em paralelo, a retomada e a ligação com o registro (`sample_id`). Consolidar em *shards* se a leitura virar gargalo.

## Consequências

- Antes da extração em massa, o piloto (mês 1) mede a taxa de detecção de mãos e o tempo por vídeo. Ele também verifica se o *holistic landmarker* da API Tasks está disponível em Python e se é preferível a três modelos separados (a verificar).
- Coordenadas *world* do MediaPipe: decidir no piloto se vale gravá-las também.
- O cache pode ficar no SSD de 120 GB (`/mnt/ssd120`) via link simbólico de `landmarks/`, se a leitura no HD for lenta (a medir).
- Mudar a versão do extrator gera um novo diretório e não sobrescreve o anterior. Experimentos registram a versão usada.
