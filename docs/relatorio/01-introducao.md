# 1. Introdução

## 1.1 Motivação

A Língua Brasileira de Sinais (Libras) é reconhecida como meio legal de comunicação e expressão no Brasil (Lei nº 10.436/2002). O reconhecimento automático de sinais pode apoiar ferramentas de acessibilidade, de ensino e de anotação de corpora.

## 1.2 Problema

Reconhecer sinais da Libras a partir de vídeo RGB, incluindo:
- **sinais estáticos**, definidos essencialmente pela configuração de mão (por exemplo, a maior parte do alfabeto datilológico);
- **sinais dinâmicos**, em que o movimento, o ponto de articulação e as expressões não-manuais são distintivos.

## 1.3 Objetivos

- **Geral:** desenvolver e avaliar um sistema de reconhecimento de sinais isolados da Libras que escale para vocabulários grandes.
- **Específicos** (a refinar após a Fase 1):
  1. Construir um pipeline reprodutível de extração de landmarks a partir de datasets públicos de Libras.
  2. Estabelecer baselines com avaliação independente de sinalizador.
  3. Comparar arquiteturas temporais (recorrentes, transformers, grafos espaço-temporais) e o efeito de cada parte do corpo (mãos, pose, face).
  4. Investigar formas de escalar o vocabulário, por exemplo a decomposição em parâmetros fonológicos.
  5. Demonstrar o reconhecimento em tempo real.

## 1.4 Antecedentes no projeto

Em 2024 foi construído um protótipo de classificação estática do alfabeto (26 letras, 1.681 amostras, MLP sobre 21 landmarks de uma mão), a partir do projeto aberto *hand-gesture-recognition-using-mediapipe*. Ele está preservado na tag `v0-prototipo-alfabeto`. As limitações desse protótipo motivaram a reformulação (ADR-0001).
