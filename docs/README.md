# Documentação

**Começando agora?** Leia a [explicação guiada](explicacao_guiada.md): o projeto inteiro, do vídeo ao modelo, para quem está entrando na área.

**Demo de webcam:** [demo.md](demo.md).

| Pasta | Para quê | Quando atualizar |
|---|---|---|
| [relatorio/](relatorio/) | Relatório vivo, base do futuro artigo. Um arquivo por capítulo. | Ao fim de cada sessão que produza algo relatável |
| [decisoes/](decisoes/) | ADRs (*Architecture Decision Records*): contexto, alternativas, decisão, consequências | Sempre que uma escolha entre alternativas for feita |
| [diario/](diario/) | Diário de bordo: o que foi feito, problemas, próximos passos | Ao fim de cada sessão |
| [datasets/](datasets/) | Uma ficha por dataset: origem, licença, estatísticas, como obter | Ao avaliar ou adotar um dataset |
| [literatura/](literatura/) | Fichamentos de artigos | Ao ler um artigo relevante |

## Decisões

| # | Título | Status |
|---|---|---|
| [0001](decisoes/0001-escopo-e-plano-inicial.md) | Escopo, plano em fases e remoção do fork | Aceita |
| [0002](decisoes/0002-landmarks-como-representacao.md) | Landmarks em sequência como representação única para sinais estáticos e dinâmicos | Proposta |
| [0003](decisoes/0003-pytorch.md) | PyTorch em vez de TensorFlow | Proposta |
| [0004](decisoes/0004-avaliacao-independente-de-sinalizador.md) | Avaliação independente de sinalizador | Proposta |
| [0005](decisoes/0005-dados-fora-do-git-e-onedrive.md) | Dados fora do git e fora do OneDrive | Proposta |
| [0006](decisoes/0006-documentacao-e-contexto-versionados.md) | Documentação e contexto do Claude versionados; uso restrito de subagentes | Proposta |
| [0007](decisoes/0007-ambiente-em-container.md) | Ambiente em container Docker (abordagem híbrida) | Proposta |
| [0008](decisoes/0008-estrategia-de-dados-inicial.md) | Estratégia de dados inicial (MINDS-Libras principal, em camadas) | Proposta |
| [0009](decisoes/0009-objetivo-poc-e-dataset-vivo.md) | Objetivo de 6 meses: POC demonstrável e base de dados extensível | Aceita |
| [0010](decisoes/0010-esquema-canonico-de-dados.md) | Esquema canônico de dados com adaptadores por fonte | Aceita |
| [0011](decisoes/0011-local-do-registro-de-dados.md) | Onde vive o registro de dados | Aceita |
| [0012](decisoes/0012-cache-de-landmarks-brutos.md) | Cache de landmarks brutos, com normalização no carregamento | Aceita |
| [0013](decisoes/0013-vocabulario-alvo-signbank.md) | Vocabulário-alvo baseado no Signbank da Libras (UFSC) | Aceita |
| [0014](decisoes/0014-plano-de-treinamento-e-melhoria-continua.md) | Plano de treinamento: sinais isolados, sequências de sinais e melhoria contínua | Aceita |
| [0015](decisoes/0015-extracao-com-recorte-guiado-pela-pose.md) | Extração de landmarks com recorte guiado pela pose (`mp_v2`) | Aceita |
| [0016](decisoes/0016-modelo-da-demo-v0-e-teste-de-webcam.md) | Modelo da demo v0, corte automático de sinais e teste de webcam | Proposta |
| [0017](decisoes/0017-coleta-do-canal-incluir-tecnologia.md) | Coleta dos vídeos do canal Incluir Tecnologia (YouTube) | Proposta |
| [0018](decisoes/0018-distribuicao-do-modelo-da-demo.md) | Distribuição do modelo da demo por GitHub Release | Proposta |
