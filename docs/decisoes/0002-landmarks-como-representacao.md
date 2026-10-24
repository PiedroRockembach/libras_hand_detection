# ADR-0002: Landmarks em sequência como representação única para sinais estáticos e dinâmicos

- **Status:** Proposta
- **Data:** 2026-10-22
- **Relacionadas:** ADR-0001

## Contexto

Um sinal da Libras combina configuração de mão, ponto de articulação, movimento, orientação da palma e expressões não-manuais. É preciso escolher a representação de entrada do modelo e decidir se sinais estáticos e dinâmicos terão tratamentos separados.

## Alternativas consideradas

1. **Pixels (RGB) com rede de vídeo** (por exemplo, I3D ou VideoMAE): melhor acurácia em alguns benchmarks, mas exige muitos dados, é sensível a fundo, iluminação e aparência do sinalizador, e custa caro em treino e inferência.
2. **Landmarks só das mãos** (como no fork): leve, mas perde ponto de articulação e expressões não-manuais.
3. **Landmarks de mãos + pose + subconjunto da face, em sequência:** leve, mais robusto a aparência e fundo, e cobre os cinco parâmetros. Depende da qualidade do estimador de pose: oclusões e mãos rápidas podem gerar *motion blur* e falhas.
4. **Modelos separados para estático e dinâmico** versus **modelo único**.

## Decisão

Usar a alternativa 3 com modelo único: todo sinal vira uma sequência `T × K × C` (quadros × pontos × coordenadas), mais uma máscara para pontos não detectados. Um sinal estático é apenas uma sequência quase constante. A extração usa a API *Tasks* do MediaPipe (hand, pose e face landmarker).

## Justificativa

- Com poucos dados por sinal, que é o cenário da Libras, landmarks generalizam melhor entre sinalizadores do que pixels.
- Um modelo único evita ter que decidir *a priori* se um sinal é estático. Várias letras do alfabeto, por exemplo, envolvem movimento.
- Tem custo compatível com uma demo em tempo real.

## Consequências

- A qualidade da extração vira o gargalo. A Fase 2 precisa medir a taxa de detecção de mãos por dataset.
- Ablações planejadas: só mãos vs. mãos+pose vs. mãos+pose+face; esquemas de normalização.
- Uma rota RGB (ou híbrida) continua sendo uma comparação possível no artigo, se houver tempo.
