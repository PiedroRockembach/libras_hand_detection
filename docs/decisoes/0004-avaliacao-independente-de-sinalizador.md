# ADR-0004: Avaliação independente de sinalizador

- **Status:** Proposta
- **Data:** 2026-10-22

## Contexto

Datasets de sinais costumam ter poucos sinalizadores, cada um repetindo os mesmos sinais. Uma divisão aleatória de amostras coloca a mesma pessoa no treino e no teste. O modelo então aprende traços da pessoa e a acurácia fica inflada.

## Alternativas consideradas

1. **Divisão aleatória por amostra:** simples, mas os números ficam otimistas e não se comparam a trabalhos rigorosos.
2. **Independente de sinalizador:** divisão fixa por pessoa ou *leave-one-signer-out* (LOSO).
3. **Protocolo oficial do dataset**, quando existir.

## Decisão

- A avaliação principal é **independente de sinalizador**. Usar o protocolo oficial do dataset se ele for independente de sinalizador; caso contrário, LOSO com média ± desvio-padrão entre as dobras.
- Para comparar com trabalhos que usaram divisão aleatória, ela pode ser reportada como resultado **secundário**, identificado como tal.
- Métricas: acurácia top-1 e top-5, F1-macro e matriz de confusão.
- A validação (seleção de hiperparâmetros) também é feita com sinalizadores diferentes dos do teste.

## Consequências

- O LOSO multiplica o custo de treino pelo número de sinalizadores. Isso é viável com modelos de landmarks na 5090.
- As divisões devem ser geradas por script, com seed, e salvas, para que todo experimento use exatamente as mesmas.
