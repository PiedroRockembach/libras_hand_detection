# ADR-0008: Estratégia de dados inicial

- **Status:** Proposta
- **Data:** 2026-10-22
- **Relacionadas:** ADR-0001, ADR-0004; fichas em [docs/datasets/](../datasets/README.md)

## Contexto

O levantamento da Fase 1 mostrou que **nenhum dataset de Libras atende sozinho** a todos os requisitos: licença aberta, avaliação independente de sinalizador, vocabulário grande, sinais estáticos e dinâmicos, e diversidade de gravação.

| Dataset | A favor | Contra |
|---|---|---|
| MINDS-Libras | CC BY 4.0, download imediato, 12 sinalizadores, resultados LOPO com MediaPipe para comparar (dos Santos 2025: 0,94 ± 0,04) | 20 sinais; ambiente homogêneo (generaliza mal para o MALTA-LIBRAS) |
| LIBRAS-UFOP | 56 sinais em pares mínimos por parâmetro, o que permite analisar configuração de mão, movimento, ponto de articulação e expressão facial | 5 sinalizadores; acesso exige acordo assinado |
| V-LIBRASIL | 1.364 termos | 3 articuladores, 1 amostra cada; vídeos em câmera lenta e com legenda gravada; não redistribuível |
| MALTA-LIBRAS (PUCRS) | Fontes variadas; teste fora de domínio; grupo da mesma universidade | Vocabulário e licença das fontes a verificar |
| ASL Citizen | 2.731 sinais, 52 sinalizadores, protocolo independente de sinalizador | Outra língua; licença não comercial e sem redistribuição |

## Alternativas consideradas

1. **Só MINDS-Libras:** começa rápido, mas os resultados não sustentam afirmações de generalização.
2. **Começar pelo V-LIBRASIL** (vocabulário grande): poucas amostras por sinal e problemas de formato tornam esse dataset ruim para um primeiro baseline.
3. **Estratégia em camadas:** um dataset principal aberto para o baseline, um segundo dataset de Libras para análise fonológica, um teste externo e um dataset grande de outra língua para pré-treino, cada um entrando em uma fase.

## Decisão

Estratégia em camadas:

| Papel | Dataset | Fase | Ação imediata |
|---|---|---|---|
| Principal (baseline, ablações) | MINDS-Libras | 2–3 | Baixar (cerca de 65 GB) e contar o número efetivo de amostras |
| Segundo dataset de Libras / análise por parâmetro | LIBRAS-UFOP | 3–4 | **Assinar e enviar o acordo de uso agora** (os contatos estão na ficha) |
| Teste fora de domínio | MALTA-LIBRAS | 4 | Contatar o grupo de Barros e Kupssinskü (PUCRS) |
| Pré-treino / transferência | ASL Citizen (PopSign como alternativa CC BY) | 4 | — |
| Escala de vocabulário (poucos exemplos) | V-LIBRASIL; dicionários (Signbank, INES) | Posterior | Verificar licenças |

O protocolo segue o ADR-0004: *nested* LOPO, o mesmo de Alves 2024 e dos Santos 2025, para manter a comparabilidade.

## Justificativa

- O MINDS-Libras é o único dataset de Libras que permite começar hoje com licença aberta e ter comparação direta com o estado da arte.
- Os pares mínimos do LIBRAS-UFOP servem diretamente à direção de pesquisa fonológica (ADR-0001).
- O teste no MALTA-LIBRAS responde à principal crítica aos resultados no MINDS, que é a generalização.

## Consequências

- **Alfabeto e letras com movimento:** não há dataset público adequado. Fica pendente uma decisão própria; as opções são pedir os dados de Lima et al. 2019 (UFPB), usar as letras do V-LIBRASIL ou fazer uma coleta própria com aprovação do CEP.
- A licença do ASL Citizen proíbe redistribuição: landmarks derivados dele não podem ser publicados.
- O espaço de armazenamento precisa comportar pelo menos o MINDS (cerca de 65 GB de vídeo) mais o cache de landmarks.
