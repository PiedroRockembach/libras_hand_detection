# ADR-0009: Objetivo de 6 meses: POC demonstrável e base de dados extensível

- **Status:** Aceita (aprovada pelo pesquisador em 2026-10-24)
- **Data:** 2026-10-24
- **Relacionadas:** ADR-0001 (fases), ADR-0008 (estratégia de dados), ADR-0010 a ADR-0013

## Contexto

Em 2026-10-24 o pesquisador detalhou o objetivo do projeto para um horizonte de cerca de 6 meses:

1. construir um **modelo inicial que funcione como prova de conceito (POC)**;
2. apresentá-lo para **conseguir recursos**;
3. no futuro distante, firmar **colaborações com instituições** para gravar novos vídeos e completar o dataset.

Daí vêm dois requisitos. A estrutura tem de permitir extrair dados e montar algo agora, e tem de aceitar dados novos no futuro. Também deve gerar um levantamento do que falta (vocabulário e diversidade de gravação) para entregar às instituições parceiras.

Datasets de Libras de acesso aberto baixados em 2026-10-23/24 (script `scripts/download_libras_datasets.py`): MINDS-Libras (RGB e RGB-D), LIBRAS-EQT-UECE v4, MALTA-LIBRAS, LSWH100, Libras Movement (UCI) e Alfabeto (Passos 2020).

## Alternativas consideradas

1. **Foco em artigo sobre o MINDS-Libras:** caminho mais curto até um resultado, mas o MINDS tem 20 sinais e ambiente controlado. Outro resultado sobre ele dificilmente se sustenta como contribuição e não convence numa apresentação.
2. **Foco em pesquisa sobre generalização entre datasets, sem demo:** pergunta clara e viável, mas não produz algo demonstrável para conseguir recursos.
3. **POC demonstrável + base de dados extensível:** demo de webcam cedo. A generalização entre datasets passa a ser requisito da demo, já que o MINDS foi gravado em estúdio e a webcam é outro domínio. Os dados ficam num formato canônico próprio, preparado para receber fontes novas.
4. **Coleta de vídeos da web por raspagem (YouTube, Signbank UFSC):** aumentaria o vocabulário. Foi **descartada por ora** pelo pesquisador (2026-10-24), por questões de termos de uso, direito de imagem e rótulos ruidosos.

## Decisão

Alternativa 3, com o cronograma abaixo:

| Mês | Entrega |
|---|---|
| 1 | Esquema canônico (ADR-0010), adaptadores MINDS e EQT-UECE, inventário dos dados, verificação do formato do MALTA-LIBRAS; extrator de landmarks (ADR-0012) e piloto em 1 sinalizador |
| 2 | Extração, divisões LOSO, baseline (MLP → GRU); **demo de webcam v0** com os 20 sinais do MINDS |
| 3 | Generalização: treino com MINDS + EQT-UECE (+ MALTA), normalização e aumento de dados; teste da demo em condições reais |
| 4 | Relatório de lacunas v1 (ADR-0013), formato de submissão para parceiros, rascunho de protocolo de gravação e termo de consentimento (base para o CEP) |
| 5 | Modelo melhor (Transformer / ST-GCN), demo v1, material de apresentação |
| 6 | Folga; artigo curto opcional (registro + avaliação entre datasets) |

Papel de cada dataset:

| Dataset | Papel |
|---|---|
| MINDS-Libras | Principal: baseline, LOSO, vocabulário da demo v0 |
| LIBRAS-EQT-UECE v4 | Segundo dataset de treino/teste. Landmarks reextraídos do RGB pelo nosso pipeline; os landmarks fornecidos servem só para comparação |
| MALTA-LIBRAS | Teste fora de domínio, se o formato dos arquivos `.pt` permitir (a verificar no mês 1) |
| LSWH100 | Tarefa auxiliar de configuração de mão (posterior) |
| Alfabeto (Passos 2020) | Baixa prioridade (imagens derivadas de ASL; licença de origem a verificar) |
| Libras Movement (UCI) | Fora do pipeline (só trajetórias, sem vídeo) |
| LIBRAS-UFOP | Enviar o acordo de uso; entra quando liberado |

## Justificativa

- A demo é o que viabiliza recursos e parcerias. Antecipá-la também expõe cedo a falha de generalização para webcam, que é o principal risco técnico.
- O formato canônico com adaptadores (ADR-0010) faz com que um dataset público e a entrega de um parceiro entrem pelo mesmo caminho.
- A avaliação entre datasets continua dando material para um artigo, sem ser o objetivo principal.

## Consequências

- Reordena as fases do ADR-0001: uma versão mínima da Fase 5 (demo) entra no mês 2. A Fase 6 (coleta própria) fica restrita à preparação do protocolo e do consentimento.
- Atualiza os papéis do ADR-0008: MALTA-LIBRAS já baixado; EQT-UECE e LSWH100 incluídos; UCI fora. ASL Citizen como pré-treino continua possível, mas não foi baixado (decisão do pesquisador: só Libras por ora).
- Riscos de licença: o EQT-UECE v4 é CC BY-NC 4.0. Se a captação de recursos envolver empresa, seu uso na demo precisa ser verificado.
- Revisar este cronograma ao fim do mês 2, com o piloto e a demo v0 em mãos.
