# ADR-0016: Modelo da demo v0, corte automático de sinais e teste de webcam

- **Status:** Proposta
- **Data:** 2026-10-31
- **Relacionadas:** ADR-0004, ADR-0009, ADR-0014, ADR-0015; resultados no [relatório §4.2](../relatorio/04-experimentos-e-resultados.md)

## Contexto

A Fase A (ADR-0014) terminou com a configuração `a6` (GRU + espelhamento + vista Kinect no treino): LOSO no MINDS com top-1 de 0,963 ± 0,025 e MALTA com 0,729 ± 0,025. A demo v0 (ADR-0009) precisa de:

1. **um único modelo.** O LOSO produz 12, cada um sem um sinalizador;
2. **uma forma de achar o início e o fim de cada sinal no fluxo da câmera.** O modelo foi treinado com clipes de um sinal só, que começam e terminam em repouso;
3. **um número de desempenho fora do estúdio sem viés.** O MALTA foi usado para escolher entre a1–a6 e deixou de ser um teste neutro.

## Alternativas consideradas

**Modelo da demo:**

1. **Um modelo de uma das dobras do LOSO:** já treinado, mas deixa de fora um sinalizador.
2. **Média das previsões dos 12 modelos (*ensemble*):** mais robusto, mas 12 vezes mais caro em tempo real.
3. **Retreinar com os 12 sinalizadores, parando numa época fixa:** usa todos os dados. Sem sinalizador de validação, a época vem do LOSO.

**Corte dos sinais:**

1. **Manual (tecla):** simples, mas a demo deixa de ser natural.
2. **Janela deslizante classificada continuamente:** sem regra de início e fim, mas gera previsões espúrias nas transições e no repouso.
3. **Atividade do braço:** um sinal começa quando um punho sobe ao espaço de sinalização e termina quando os braços voltam ao repouso, com alguns quadros de contexto, como nos clipes do MINDS.

**Teste fora do estúdio:**

1. **Continuar com o MALTA:** já disponível, mas com viés de seleção.
2. **Gravar um conjunto de webcam (`webcam_v0`)** com os 20 sinais, pessoas novas e câmeras domésticas, usado só como teste.

## Decisão

- **Modelo:** alternativa 3. O a6 é retreinado com os 12 sinalizadores e para na **época 24**, a mediana das melhores épocas das 12 dobras, mantendo o calendário de taxa de aprendizado de 80 épocas. Execução: `experiments/a6_gru_mirror_kinect/final-20261031T174312Z-770f082`. O desempenho esperado é a estimativa do LOSO.
- **Corte:** alternativa 3, com a 1 como recurso (tecla espaço). Considera-se que o braço está levantado quando um punho com visibilidade > 0,5 fica acima da altura dos ombros + 1,2 largura de ombros. O sinal começa depois de 3 quadros com o braço levantado e termina depois de 10 em repouso, com 10 quadros de contexto de cada lado, mínimo de 12 e máximo de 240 quadros (`libras.demo.stream.Segmenter`).
- **Teste:** alternativa 2. `python -m libras.demo record` grava cada sinal com um identificador pseudônimo, consentimento obrigatório e registro em `PROVENANCE.jsonl`. **O conjunto não é usado para escolher modelos nem configurações.**

## Justificativa

- O retreino com todos os dados é a prática padrão depois da validação cruzada. Parar pela mediana do LOSO evita escolher a época olhando um teste.
- No MINDS, o corte por atividade do braço segue o formato dos clipes de treino (repouso → sinal → repouso). A referência é a altura dos ombros, e não o ponto médio ombro–quadril da métrica de extração, porque numa mesa o quadril costuma ficar fora da imagem.
- Verificação do pipeline em fluxo: o modelo da dobra 0 do a6, em 6 vídeos do sinalizador 01 que essa dobra não viu, acertou 6 de 6, tanto com o vídeo inteiro quanto com o corte automático. Encontrou exatamente um sinal por vídeo. É uma verificação, não uma métrica.
- Um teste de webcam nunca usado na escolha dá o número que importa para a demo e para o artigo.

## Consequências

- **Limites da demo v0:** 20 sinais; um sinal por vez, começando e terminando em repouso; vocabulário fechado, de modo que qualquer gesto vira um dos 20 sinais (não há rejeição de "sinal desconhecido"). A sinalização contínua fica para a Fase C (CTC).
- **Parâmetros do corte** (limiar de 1,2 largura de ombros, 3/10 quadros) foram escolhidos por coerência com os clipes do MINDS, sem ajuste com dados de webcam. Precisam ser revistos com as primeiras gravações.
- **Pendências:** termo de consentimento para as gravações (a verificar com o laboratório); adaptador `webcam_v0` no registro (ADR-0010); avaliação dos modelos do LOSO no `webcam_v0`; limiar de rejeição (ADR-0014, em aberto).
- **Ambiente:** a demo roda fora do container (ADR-0007) e precisa de `uv` e de uma webcam no host.
