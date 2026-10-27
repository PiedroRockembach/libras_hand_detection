# ADR-0013: Vocabulário-alvo baseado no Signbank da Libras (UFSC)

- **Status:** Aceita (aprovada pelo pesquisador em 2026-10-24)
- **Data:** 2026-10-24
- **Relacionadas:** ADR-0009, ADR-0010, ADR-0011; ficha [signbank-libras-ufsc.md](../datasets/signbank-libras-ufsc.md)

## Contexto

O ADR-0009 prevê um relatório de lacunas para entregar a instituições parceiras: quais sinais faltam, quais têm poucos sinalizadores e que diversidade de gravação falta. Para medir o que falta, é preciso um **vocabulário-alvo** (o universo de sinais desejado), comparado com o que o registro (ADR-0010) já cobre.

O Signbank da Libras (UFSC) é um banco lexical com aspectos fonológicos. Segundo fonte secundária, tem 3.118 sinais (a verificar). Em consulta de 2026-10-24: o portal busca os dados numa API separada, não publica termos de uso, e a política de privacidade publicada é um texto provisório. Não há base clara para uso automatizado.

## Alternativas consideradas

1. **Sem vocabulário-alvo:** o relatório mede só a cobertura do que já existe. Não responde "o que falta gravar".
2. **Lista temática montada manualmente** (por exemplo, saúde, educação, cotidiano): rápida e alinhada a uma aplicação, mas arbitrária e sem ligação com um léxico de referência.
3. **Signbank UFSC como base do vocabulário-alvo, obtido por pedido ao grupo responsável:** léxico de referência acadêmico, possivelmente com anotação fonológica, e o contato abre uma primeira parceria.
4. **Raspar o Signbank:** descartada pelo pesquisador em 2026-10-24 (ADR-0009, alternativa 4).

## Decisão

Alternativa 3, com a 2 como complemento:

- Pedir ao grupo do Signbank (UFSC / Levante Lab) a lista de sinais com identificadores e, se houver, as anotações fonológicas.
- O vocabulário-alvo é uma lista de `gloss_id` ligada aos sinais do Signbank (`signbank_ref` na tabela `glosses`, ADR-0010). Subconjuntos temáticos (alternativa 2) são marcados sobre essa lista.
- Enquanto a licença não for esclarecida, a lista derivada do Signbank fica fora do git, em `LIBRAS_DATA_ROOT/registry/` (ADR-0011).
- Se não houver resposta, montar manualmente uma lista temática pequena e ligá-la ao Signbank por consulta individual no portal.

## Justificativa

- Um léxico de referência dá legitimidade ao relatório de lacunas diante de instituições parceiras.
- Pedir em vez de raspar respeita termos que ainda não estão definidos e transforma a necessidade de dados num primeiro contato institucional.
- Se as anotações fonológicas existirem, elas viabilizam a decomposição por parâmetros (ADR-0001) sem nova coleta.

## Consequências

- Ação imediata: redigir e enviar o pedido ao grupo do Signbank. A resposta pode demorar, mas o relatório de lacunas só é necessário no mês 4.
- O mapeamento dos rótulos do MINDS, EQT-UECE e MALTA para o Signbank é feito à mão e precisa de revisão de alguém fluente em Libras (variantes regionais, sinais homônimos).
- Atualizar a ficha do Signbank com a resposta (licença, campos disponíveis).
