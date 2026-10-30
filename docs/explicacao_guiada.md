# Explicação guiada do projeto

*Para quem está chegando agora na área de reconhecimento de língua de sinais ou de aprendizado de máquina com vídeo. Escrito em 2026-10-29. Este texto explica o **porquê** das escolhas; os detalhes formais estão nos ADRs ([docs/decisoes/](decisoes/)) e no relatório ([docs/relatorio/](relatorio/)). Quando um número aparece aqui, a fonte está indicada; o que ainda não foi medido está marcado como "a verificar" ou "a medir".*

---

## Sumário

1. [O problema em uma frase](#1-o-problema-em-uma-frase)
2. [O que é um sinal, do ponto de vista da máquina](#2-o-que-é-um-sinal-do-ponto-de-vista-da-máquina)
3. [Vídeo, não imagem: por que o tempo importa](#3-vídeo-não-imagem-por-que-o-tempo-importa)
4. [Pixels ou esqueleto? A escolha da representação](#4-pixels-ou-esqueleto-a-escolha-da-representação)
5. [Os dados que temos](#5-os-dados-que-temos)
6. [Do vídeo aos landmarks: a extração](#6-do-vídeo-aos-landmarks-a-extração)
7. [Dos landmarks à entrada do modelo: o pré-processamento](#7-dos-landmarks-à-entrada-do-modelo-o-pré-processamento)
8. [O modelo](#8-o-modelo)
9. [Como o modelo aprende: o treino](#9-como-o-modelo-aprende-o-treino)
10. [Como sabemos se funciona: a avaliação](#10-como-sabemos-se-funciona-a-avaliação)
11. [De sinais isolados a frases](#11-de-sinais-isolados-a-frases)
12. [Como o modelo cresce com vídeos novos](#12-como-o-modelo-cresce-com-vídeos-novos)
13. [Reprodutibilidade: por que tanto registro](#13-reprodutibilidade-por-que-tanto-registro)
14. [Limitações que precisam ficar claras](#14-limitações-que-precisam-ficar-claras)
15. [A pesquisa em que nos baseamos](#15-a-pesquisa-em-que-nos-baseamos)
16. [Glossário](#16-glossário)
17. [Onde está cada coisa no repositório](#17-onde-está-cada-coisa-no-repositório)

---

## 1. O problema em uma frase

Dado um vídeo de uma pessoa sinalizando em Libras, **dizer qual sinal (ou qual sequência de sinais) ela fez**.

O objetivo de longo prazo é um tradutor. O caminho até lá é dividido em etapas (ADR-0014):

1. **Sinais isolados:** o vídeo contém um único sinal (ex.: "AMARELO"). O modelo escolhe uma resposta entre um vocabulário fechado. É um problema de **classificação**.
2. **Sequências de sinais:** o vídeo contém uma frase (ex.: "EU QUERER PAGAR CONTA"). O modelo precisa dizer quais sinais apareceram **e em que ordem**, sem que ninguém diga onde um termina e o outro começa. Isso se chama **reconhecimento contínuo**.
3. **Tradução:** transformar a sequência de sinais em uma frase correta em português. Libras tem gramática própria, então não basta trocar palavra por palavra. Esta etapa fica para depois dos 6 meses.

Este documento foca nas etapas 1 e 2, que são o que vamos treinar agora.

---

## 2. O que é um sinal, do ponto de vista da máquina

Na linguística das línguas de sinais, um sinal é descrito por **parâmetros** (análogos aos fonemas de uma língua falada):

| Parâmetro | O que é | O que a máquina precisa enxergar |
|---|---|---|
| Configuração de mão | o formato da mão (dedos abertos, fechados, em "L"...) | a posição de cada dedo |
| Ponto de articulação | onde o sinal é feito (testa, queixo, peito, espaço neutro) | a posição da mão **em relação ao corpo e ao rosto** |
| Movimento | o trajeto da mão (reto, circular, repetido) | como a posição muda **ao longo do tempo** |
| Orientação da palma | para onde a palma aponta | a geometria 3D da mão |
| Expressões não-manuais | expressão facial, movimento da cabeça, boca | o rosto |

Duas consequências práticas:

- **Olhar só para as mãos não basta.** Sinais diferentes podem ter a mesma configuração de mão e diferir só no local ou na expressão facial. Por isso extraímos **mãos, pose do corpo e rosto** (ADR-0002).
- **Olhar só para um quadro não basta.** O movimento é um parâmetro. Isso nos leva à próxima seção.

---

## 3. Vídeo, não imagem: por que o tempo importa

Um projeto comum para iniciantes é reconhecer o **alfabeto manual** a partir de **uma foto** da mão. Isso funciona para letras estáticas, mas não para a Libras real:

- a maioria dos sinais tem **movimento**: o sinal é o trajeto, não uma pose;
- até no alfabeto há letras com movimento (ex.: H, J, K, X, Y, Z, Ç);
- alguns sinais têm a mesma forma de mão no começo e só se distinguem pelo que acontece depois.

Por isso, a unidade de entrada do nosso modelo **não é uma imagem, e sim uma sequência de quadros**. Concretamente:

```
vídeo  →  sequência de T quadros  →  para cada quadro, K pontos do corpo com C coordenadas
                                     (um "esqueleto" por quadro)
       →  tensor de forma  T × K × C
```

- **T** = número de quadros (tempo). Um vídeo de 3 s a 30 fps tem T = 90.
- **K** = número de pontos (dedos, punhos, ombros, lábios...).
- **C** = coordenadas de cada ponto (x, y e, às vezes, profundidade z).

**E os sinais estáticos?** Não precisam de tratamento especial. Um sinal estático é só uma sequência em que os pontos quase não mudam. Usar **um único modelo** para os dois casos evita ter que decidir antes se um sinal é estático ou dinâmico (ADR-0002).

**O que o modelo precisa fazer com o tempo:** combinar a informação de muitos quadros. Há várias famílias de redes neurais que fazem isso (seção 8): redes recorrentes (que leem um quadro de cada vez, guardando uma memória), convoluções temporais (que olham janelas de quadros vizinhos) e *transformers* (que comparam todos os quadros entre si).

**Uma diferença importante em relação a imagens:** vídeos têm **comprimentos diferentes**. Um mesmo sinal pode durar 2 s para uma pessoa e 4 s para outra. Para sinais isolados, **reamostramos** cada sequência para um T fixo (64 quadros na Fase A). Para frases, mantemos o comprimento variável e usamos uma função de perda própria para sequências (CTC, seção 11).

---

## 4. Pixels ou esqueleto? A escolha da representação

Existem dois grandes caminhos para dar um vídeo a uma rede neural:

### Caminho 1: pixels (RGB)

A rede recebe os quadros do vídeo diretamente. Exemplos de modelos: I3D, VideoMAE, TimeSformer.

- **Vantagem:** vê tudo, inclusive detalhes que um estimador de pose pode perder (contato entre mãos, mão na frente do rosto).
- **Desvantagens:** precisa de **muitos** dados para não decorar o fundo, a roupa, a iluminação e a aparência de cada pessoa; o treino e a inferência são caros.

### Caminho 2: esqueleto (landmarks)

Primeiro, um modelo pronto (o **MediaPipe**, do Google) encontra os **pontos-chave** do corpo em cada quadro: 21 pontos por mão, 33 do corpo e 478 do rosto. Depois, a nossa rede recebe **só as coordenadas desses pontos**.

- **Vantagens:** a entrada é pequena (centenas de números por quadro, contra milhões de pixels); é naturalmente independente de fundo, roupa, cor de pele e iluminação; o modelo treina rápido e roda em tempo real num notebook.
- **Desvantagens:** se o MediaPipe errar (mão borrada, mãos se tocando), o erro passa adiante e o modelo não tem como recuperar.

### Por que escolhemos landmarks (ADR-0002)

1. **Temos poucos dados por sinal.** Os datasets públicos de Libras têm de 5 a 12 sinalizadores. Com tão pouca variedade, um modelo de pixels tende a aprender "quem é a pessoa" em vez de "qual é o sinal". Com landmarks, a aparência da pessoa simplesmente não está na entrada.
2. **A literatura recente mostra que pose bem extraída compete com pixels.** Trocar só o estimador de pose para o MediaPipe elevou o modelo SPOTER de 63,18% para 78,29% no WLASL100 ([Bohacek 2022](literatura/bohacek-2022-spoter-mediapipe.md)); o Uni-Sign mostrou pose igualando RGB ([Li 2025](literatura/li-2025-uni-sign.md)); o primeiro lugar da competição Kaggle de sinais isolados (2023) usou só landmarks do MediaPipe ([Sohn 2023](literatura/sohn-2023-kaggle-islr.md)).
3. **Em Libras, há trabalhos com landmarks com boa acurácia** no protocolo mais rigoroso: [dos Santos 2025](literatura/santos-2025-landmark-subset.md) relata 0,94 ± 0,04 no MINDS-Libras com MediaPipe.
4. **Tempo real e custo:** a meta inclui uma demo de webcam. Landmarks permitem isso sem GPU dedicada.

**Sendo honestos com o outro lado:** em Libras, um I3D com pixels chegou a 92,5% no MINDS ([Costa 2025](literatura/costa-2025-i3d-libras.md)), e os mesmos autores mostram que, no mesmo protocolo, RGB ficou acima de landmarks ([Costa 2026](literatura/costa-2026-distilacao-mamba.md)). Landmarks também perdem informação de ponto de articulação em relação a pixels ([Yin 2026](literatura/yin-2026-phonological-perception.md)). Uma comparação com RGB continua prevista como trabalho futuro (ADR-0002).

---

## 5. Os dados que temos

Inventário completo em [relatório §3.2.1](relatorio/03-metodologia.md) e fichas em [docs/datasets/](datasets/).

| Dataset | O que tem | Para que usamos |
|---|---|---|
| **MINDS-Libras** (UFMG) | 20 sinais isolados, 12 sinalizadores, ~5 repetições cada, gravado em estúdio com duas câmeras (Canon e Kinect) | Base do primeiro modelo (Fase A) e da avaliação principal |
| **LIBRAS-EQT-UECE** | 44 sinais isolados + **131 frases**, 5 sinalizadores, domínio de atendimento de uma distribuidora de energia | Ampliar o vocabulário (Fase B) e treinar o reconhecimento de **frases** (Fase C) |
| **MALTA-LIBRAS** (PUCRS) | Clipes de fontes variadas (pessoas, fundos e câmeras diferentes) | Teste **fora de domínio**: o modelo funciona fora do estúdio onde foi treinado? |

Todos passam por um **registro canônico** (ADR-0010): cada vídeo vira uma linha com identificador único, sinal (glosa), sinalizador, licença, câmera etc. Isso permite juntar datasets com formatos diferentes e, no futuro, receber vídeos de instituições parceiras no mesmo formato.

**Glosa** é o nome convencional de um sinal, escrito em maiúsculas em português (ex.: AMARELO, BANCO). É um rótulo, não uma tradução.

---

## 6. Do vídeo aos landmarks: a extração

Esta etapa roda **uma única vez** por vídeo e já foi concluída para todos os dados (2026-10-29). Comando: `scripts/extract_landmarks.sh`.

### 6.1 Três modelos do MediaPipe

Usamos a API *Tasks* do MediaPipe com três modelos separados:

- **Pose** (33 pontos do corpo: rosto aproximado, ombros, cotovelos, punhos, quadril...);
- **Mãos** (21 pontos por mão: punho e 4 articulações por dedo), até 2 mãos;
- **Rosto** (478 pontos: contorno, olhos, sobrancelhas, lábios...).

Os arquivos de modelo estão fixados por URL e por hash SHA-256 no manifesto da extração, para que qualquer pessoa reproduza exatamente os mesmos pontos.

### 6.2 O problema da pessoa pequena no quadro e a solução com recortes (ADR-0015)

Os vídeos do MINDS e do EQT são em 1920×1080, mas a pessoa ocupa uma parte pequena da imagem. No piloto, rodando os detectores no quadro inteiro (`mp_v1`), **o rosto não foi detectado em nenhum quadro do MINDS** e só em 14% do EQT.

A solução (`mp_v2`, a versão em uso):

1. roda a **pose** no quadro inteiro (ela funciona bem mesmo com a pessoa pequena);
2. usa a pose para definir duas **regiões de interesse fixas** para cada vídeo: o **espaço de sinalização** (onde as mãos se movem, com margem de 25%) e a **cabeça**;
3. roda os detectores de **mãos** e de **rosto** só dentro desses recortes, onde a pessoa aparece maior;
4. converte as coordenadas de volta para o quadro inteiro.

Resultado do piloto: rosto de 0% → 100% no MINDS; perda de mãos durante o sinal caiu no MINDS e no EQT (tabela completa no ADR-0015).

### 6.3 Qualidade medida na extração completa

A métrica principal é a **mão perdida durante o sinal**: em quadros em que a pose vê o punho acima da linha média entre ombros e quadril (ou seja, a pessoa está sinalizando), em que fração nenhuma mão foi detectada perto dele. Resultado da extração completa (`python -m libras.extract report --config configs/extraction/mp_v2.yaml`, 2026-10-29):

| Fonte | Vídeos | Mão perdida durante o sinal |
|---|---|---|
| LIBRAS-EQT-UECE | 5.263 | 5,2% |
| MINDS Canon | 1.158 | 9,1% |
| MINDS Kinect | 1.200 | 16,4% |
| MALTA (clipes com sinais do MINDS/EQT) | 293 | 19,2% |

Pose e rosto foram detectados em praticamente todos os quadros.

### 6.4 O que fica gravado (ADR-0012)

Um arquivo `.npz` por vídeo, em `LIBRAS_DATA_ROOT/landmarks/mp_v2/<fonte>/`, com **todos** os pontos brutos (sem seleção, sem normalização), no fps original, mais **máscaras de presença** (`hand_present`, `face_present`...). Quando um detector falha num quadro, as coordenadas ficam como `NaN` ("não é um número").

**Por que gravar o bruto?** Porque normalização, escolha de pontos e reamostragem são decisões que queremos **testar e mudar**. Se gravássemos já processado, cada teste exigiria reextrair horas de vídeo. Assim, essas escolhas viram parâmetros num arquivo YAML.

---

## 7. Dos landmarks à entrada do modelo: o pré-processamento

Esta etapa roda **a cada treino**, ao carregar os dados, e é configurada no YAML do experimento (`configs/experiments/`). É aqui que a maior parte do "conhecimento do domínio" entra. A literatura indica que **a normalização é o fator de maior impacto** em modelos de landmarks ([De Coster 2023](literatura/decoster-2023-sign-embeddings.md); [Zelezny 2025](literatura/zelezny-2025-pose-ablation.md)).

### 7.1 Seleção de pontos

Não usamos os 543 pontos. Em Libras, usar todos piora muito o resultado ([dos Santos 2025](literatura/santos-2025-landmark-subset.md)), porque o modelo se perde em centenas de pontos do contorno do rosto que pouco informam. A seleção inicial da Fase A:

- **mãos:** os 2 × 21 pontos (forma da mão);
- **pose:** nariz, ombros, cotovelos e punhos (posição do braço em relação ao corpo);
- **rosto:** um subconjunto de lábios, sobrancelhas e ponta do nariz (expressões não-manuais e referência de local).

A seleção é uma lista de índices no YAML, e testar outras é uma das ablações previstas.

### 7.2 Correção da proporção da imagem

O MediaPipe dá coordenadas entre 0 e 1 em cada eixo. Num vídeo 16:9, uma unidade em x é maior que uma em y. Multiplicamos x pela razão largura/altura para que distâncias tenham o mesmo significado nos dois eixos.

### 7.3 Normalização pelo corpo

Pessoas diferentes ficam em posições e distâncias diferentes da câmera. Para que o modelo veja "o mesmo sinal" independentemente disso:

- **centro:** o ponto médio entre os ombros vira a origem (0, 0);
- **escala:** dividimos tudo pela largura dos ombros.

Depois disso, "a mão está na altura do queixo" vira um número parecido para qualquer pessoa, em qualquer enquadramento. Isso é especialmente importante porque cada dataset tem um enquadramento diferente, e não queremos que o modelo aprenda "este enquadramento = este dataset" (o "atalho por dataset", relatório §3.2.1).

### 7.4 Coordenadas locais da mão

Além da posição global, cada mão também é descrita **em relação ao próprio punho**, dividida pelo tamanho da mão. Isso isola a **configuração de mão** (o formato) da **posição** (onde ela está). Técnica usada por [Bohacek 2022](literatura/bohacek-2022-spoter.md) e [De Coster 2023](literatura/decoster-2023-sign-embeddings.md).

### 7.5 Qual mão é qual

O MediaPipe às vezes troca "esquerda" e "direita", ou dá o mesmo rótulo às duas mãos (ADR-0015). Ao carregar, reatribuímos cada mão detectada ao **punho da pose mais próximo**, que é mais estável.

### 7.6 Quadros sem detecção

Se a mão some por poucos quadros (borrão de movimento), **interpolamos** linearmente entre o antes e o depois. Em Libras, preencher falhas deu ganhos grandes ([dos Santos 2025](literatura/santos-2025-landmark-subset.md)). Falhas longas (ex.: braço em repouso fora do quadro) viram zero, e o modelo recebe também um **indicador de presença** da mão em cada quadro, para saber a diferença entre "mão na origem" e "mão ausente".

### 7.7 Velocidade

Além da posição, damos ao modelo a **diferença entre quadros consecutivos** (velocidade). O movimento é um parâmetro do sinal, e entregá-lo pronto facilita o aprendizado com poucos dados (receita do [Kaggle ISLR](literatura/sohn-2023-kaggle-islr.md)).

### 7.8 Reamostragem temporal

Para sinais isolados, cada sequência é reamostrada para **T = 64 quadros** (valor inicial do ADR-0014, a ajustar). Um vídeo de 120 quadros é "comprimido"; um de 40 é "esticado".

### 7.9 Aumento de dados

Com poucos exemplos, criamos variações artificiais **só no treino**, para o modelo não decorar os exemplos:

| Aumento | O que simula |
|---|---|
| escala e rotação leves | pessoas de tamanhos diferentes, câmera levemente torta (rotação forte prejudica, [Zelezny 2025](literatura/zelezny-2025-pose-ablation.md)) |
| recorte das pontas e distorção suave do ritmo | pessoas que começam, terminam e sinalizam em ritmos diferentes (com T fixo, mudar só a velocidade não teria efeito: a reamostragem desfaria) |
| ruído pequeno nas coordenadas | imprecisão do MediaPipe |
| descarte de quadros e de uma mão por um trecho | falhas de detecção |
| espelhamento (com troca das mãos) | sinalizadores canhotos; implementado, mas **desligado** no baseline, para ser testado como ablação |

O resultado de todo o pré-processamento, para cada vídeo, é uma matriz **T × F**: T quadros, F números por quadro (as *features*).

---

## 8. O modelo

### 8.1 A ideia geral (ADR-0014)

```
features (T × F)
  → codificador temporal  → uma representação por quadro (T × D)
       │
       ├── sinal isolado:  média ao longo do tempo → embedding (D) → classificador de cosseno → sinal
       │
       └── frase (Fase C): classificador de cosseno em cada quadro (+ "blank") → CTC → sequência de sinais
```

Três peças: o **codificador**, que transforma a sequência numa representação útil; a **cabeça isolada**, que decide qual sinal; e, mais tarde, a **cabeça de sequência**, que lê frases. As duas cabeças compartilham o mesmo codificador e os mesmos pesos de classificação.

### 8.2 O codificador: progressão do simples ao complexo

Começamos simples de propósito. Um modelo complexo que dá um número ruim não diz se o problema é o modelo ou os dados; um modelo simples serve de régua.

1. **MLP sobre estatísticas (sanidade):** resume cada feature ao longo do tempo (média, desvio, mínimo, máximo) e passa por uma rede densa simples. Ignora a ordem dos quadros. Se este modelo já acerta muito, os dados estão bons; se acerta pouco, algo está errado no pré-processamento. **Não é o modelo final**: ele é incapaz de distinguir sinais que diferem só na ordem do movimento.
2. **GRU (o baseline da Fase A):** uma **rede recorrente** que lê os quadros em ordem, mantendo uma "memória" (estado oculto) atualizada a cada quadro. A GRU (*Gated Recurrent Unit*) é uma variante com "portões" que decidem o que lembrar e o que esquecer. É leve, treina bem com poucos dados e produz naturalmente **uma saída por quadro**, que é o que a cabeça de frases vai precisar. Antes dela, uma camada linear projeta as features de cada quadro para a dimensão interna.
3. **Depois (Fases B–D):** *transformer* e redes de grafo espaço-temporal (ST-GCN/SL-GCN), que são o estado da arte para esqueletos ([Jiang 2021](literatura/jiang-2021-sam-slr.md); [Selvaraj 2022](literatura/selvaraj-2022-openhands.md)), e uma versão **causal** (que só olha o passado) para tempo real.

**Por que um codificador "por quadro" desde o início?** Porque o mesmo modelo treinado em sinais isolados pode depois ser ajustado para frases, sem trocar de arquitetura. O trabalho da Fase A vira o ponto de partida da Fase C (ADR-0014).

**Bidirecional ou não?** Na Fase A, a GRU lê a sequência nos dois sentidos (frente e trás), o que costuma ajudar em sinais isolados. Para tempo real (Fase D), ela precisa ser causal. Isso é uma opção no YAML.

### 8.3 A cabeça isolada: embedding e classificador de cosseno

1. **Pooling temporal:** a média das saídas de todos os quadros vira um único vetor, o **embedding** do vídeo (uma "impressão digital" do sinal).
2. **Classificador de cosseno:** cada sinal do vocabulário tem um vetor de pesos. A pontuação de um sinal é o **cosseno do ângulo** entre o embedding do vídeo e o vetor daquele sinal (multiplicado por uma escala). Ganha o sinal mais "alinhado".

**Por que cosseno, e não a camada linear comum?** Porque permite **adicionar sinais novos sem retreinar** (*weight imprinting*): o vetor de um sinal novo pode ser simplesmente a **média dos embeddings** de alguns vídeos dele (o **protótipo**). Isso é central no plano de "dataset vivo": quando um parceiro mandar vídeos de um sinal novo, ele entra na demo imediatamente; o retreino completo oficial vem depois (ADR-0014). Ideia de Qi, Brown e Lowe, *Low-Shot Learning with Imprinted Weights* (CVPR 2018; não fichado no projeto).

### 8.4 Tamanho

O baseline GRU da Fase A (`configs/experiments/a1_gru_minds.yaml`) tem cerca de 1,3 milhão de parâmetros; o número exato fica registrado no `meta.json` de cada treino. Para comparação, modelos de vídeo RGB têm dezenas de milhões. É isso que permite treinar em CPU e rodar em tempo real.

---

## 9. Como o modelo aprende: o treino

- **Função de perda:** entropia cruzada (o padrão para classificação), com *label smoothing* (não exige 100% de certeza, o que ajuda com poucos dados).
- **Otimizador:** AdamW, com taxa de aprendizado que decai em cosseno ao longo das épocas.
- **Época:** uma passada por todos os vídeos de treino. Cada época vê versões aumentadas diferentes dos mesmos vídeos.
- **Validação:** a cada época, medimos o desempenho num sinalizador **separado** (nem no treino, nem no teste) e guardamos o modelo da melhor época. Isso evita escolher o modelo olhando o teste.
- **Seed fixa:** todas as fontes de aleatoriedade (inicialização, ordem dos dados, aumentos) partem de uma semente gravada no YAML, para o treino ser repetível.
- **Hardware:** modelos de landmarks são pequenos. A Fase A roda em CPU; a GPU acelera, mas não é obrigatória. Para o container usar a GPU, o host precisa do *NVIDIA Container Toolkit*.
- **Como rodar:** `docker compose run --rm libras python -m libras.train --config configs/experiments/a1_gru_minds.yaml`. O modelo de sanidade (seção 8.2) usa `a0_stats_mlp_minds.yaml`.

---

## 10. Como sabemos se funciona: a avaliação

### 10.1 A armadilha mais comum da área

Num dataset com 12 pessoas repetindo os mesmos sinais, se sortearmos vídeos aleatoriamente para treino e teste, **a mesma pessoa aparece nos dois**. O modelo aprende o jeito daquela pessoa e a acurácia fica inflada. Na prática, um tradutor sempre vai encontrar pessoas que nunca viu.

### 10.2 Avaliação independente de sinalizador (ADR-0004)

Usamos **LOSO** (*leave-one-signer-out*, "deixa um sinalizador de fora"):

```
Para cada um dos 12 sinalizadores do MINDS:
    teste     = todos os vídeos dessa pessoa
    validação = todos os vídeos de outra pessoa (para escolher a melhor época)
    treino    = as 10 restantes
    treina um modelo do zero e mede no teste
Resultado = média ± desvio-padrão das 12 rodadas ("dobras")
```

São 12 modelos treinados por experimento. É caro, mas é o único número que reflete o uso real. Os trabalhos mais rigorosos de Libras usam um protocolo equivalente (*leave-one-person-out*; [Alves 2024](literatura/alves-2024-skeleton-image.md), [dos Santos 2025](literatura/santos-2025-landmark-subset.md)), o que torna nossos números comparáveis aos deles.

### 10.3 Métricas

- **Top-1:** fração de vídeos em que o primeiro palpite está certo.
- **Top-5:** fração em que a resposta certa está entre os 5 primeiros palpites (útil numa interface que sugere opções).
- **F1-macro:** média do F1 de cada classe; penaliza o modelo que ignora sinais difíceis.
- **Matriz de confusão:** quais sinais são confundidos com quais. Ajuda a entender o erro (ex.: sinais com a mesma configuração de mão em locais diferentes).

### 10.4 Teste fora de domínio: MALTA

Os modelos de cada dobra também são avaliados nos clipes do MALTA com os mesmos 20 sinais do MINDS (124 clipes, de pessoas, câmeras e fundos diferentes). Trabalhos anteriores mostram que modelos treinados no MINDS generalizam mal para o MALTA ([Delucis 2025](literatura/delucis-2025-malta-libras.md)). É o número que diz se a demo de webcam tem chance de funcionar fora do estúdio. Com poucos clipes, o intervalo de confiança é largo, e o resultado é reportado por subcoleção (ADR-0015).

---

## 11. De sinais isolados a frases

Requisito crítico do projeto: reconhecer **sinais em sequência**, identificando a passagem de um sinal para o outro.

### 11.1 Por que é difícil

- Numa frase, ninguém marca onde cada sinal começa e termina.
- Os sinais se influenciam: o fim de um molda o começo do próximo (**coarticulação**), então um sinal dentro de uma frase não é igual a ele isolado.

### 11.2 A solução: CTC (Fase C)

A **CTC** (*Connectionist Temporal Classification*; Graves et al., ICML 2006, não fichado no projeto) é a técnica padrão de reconhecimento contínuo, de voz a língua de sinais. A ideia:

1. o modelo dá, **para cada quadro**, uma pontuação para cada sinal do vocabulário **mais um símbolo extra, o "blank"** (= "nenhum sinal novo aqui", usado nas transições);
2. para treinar, basta saber **a sequência de sinais do vídeo** (ex.: EU QUERER PAGAR), **sem saber em que quadro cada um está**. A CTC soma todas as formas possíveis de alinhar essa sequência aos quadros;
3. na inferência, juntamos os quadros consecutivos com o mesmo sinal e removemos os blanks: `blank blank EU EU blank QUERER QUERER blank PAGAR` → `EU QUERER PAGAR`.

As **fronteiras entre sinais emergem sozinhas**: o modelo aprende a colocar o blank nas transições.

Como o codificador já produz uma saída por quadro (seção 8.2), a cabeça CTC usa o mesmo classificador de cosseno aplicado quadro a quadro. Um sinal novo adicionado por protótipo fica disponível nas duas cabeças.

### 11.3 De onde vêm as frases para treinar

1. **As 131 frases do EQT-UECE**, anotadas com a sequência de glosas em Libras por uma pessoa fluente. O dataset traz só o texto em português, que não corresponde à ordem dos sinais. Como cada frase foi gravada ~30 vezes, **131 anotações rotulam ~3.900 vídeos**. O kit de anotação já foi entregue (`docs/anotacao/`).
2. **Frases sintéticas:** concatenar landmarks de sinais isolados com transições interpoladas. Ensinam fronteiras, mas não têm coarticulação real; servem só como aumento de dados, nunca como teste.
3. **Mais tarde:** Corpus de Libras da UFSC (acesso a pedir) e gravações de parceiros.

### 11.4 Como medir: WER

Para frases, a métrica é a **WER** (*word error rate*): quantas substituições, inserções e remoções são necessárias para transformar a sequência prevista na correta, dividido pelo tamanho da correta. Também é independente de sinalizador (LOSO nos 5 sinalizadores do EQT).

---

## 12. Como o modelo cresce com vídeos novos

O projeto foi desenhado para receber vídeos de instituições parceiras no futuro (ADR-0009, ADR-0014):

```
vídeos novos → adaptador → registro → snapshot novo (vN+1, imutável)
  → extração só dos vídeos novos → controle de qualidade → validação dos rótulos por pessoa fluente
  → (rápido)  protótipos dos sinais novos entram na demo, sem retreino
  → (oficial) retreino completo do zero no snapshot vN+1, mesma configuração
  → avaliação nos conjuntos de teste fixos → o modelo novo só substitui o antigo se não piorar
```

**Por que retreinar do zero, e não continuar treinando o modelo antigo?** Treinar só com dados novos faz o modelo **esquecer** o que sabia (*esquecimento catastrófico*), e o resultado depende da ordem em que os dados chegaram, o que impede a reprodução. O retreino completo é barato com landmarks.

---

## 13. Reprodutibilidade: por que tanto registro

Em pesquisa, um número que ninguém consegue reproduzir não vale. Regras do projeto (CLAUDE.md):

- todo experimento roda a partir de um **YAML** em `configs/`, com **seed fixa**, e grava o **commit** do código;
- toda alteração nos dados roda por **script versionado** e grava uma linha em `PROVENANCE.jsonl`;
- nenhum número é reportado sem estar num **log de experimento**;
- os resultados de cada treino ficam em `LIBRAS_DATA_ROOT/experiments/<experimento>/<execução>/`, com a configuração usada, o commit, as métricas por dobra e o resumo.

---

## 14. Limitações que precisam ficar claras

- **Vocabulário pequeno:** a POC reconhece os 20 sinais do MINDS e, depois, os 44 do EQT. Libras tem milhares de sinais.
- **Poucas pessoas:** 12 + 5 sinalizadores. Diversidade (idade, região, estilo) é limitada.
- **Domínio fechado nas frases:** as 131 frases são todas sobre atendimento de energia, com 5 pessoas. Reconhecer frases desse domínio **não** é um tradutor geral.
- **Estúdio vs. mundo real:** MINDS e EQT foram gravados em fundo controlado. O teste no MALTA mede essa diferença.
- **Erros do MediaPipe passam adiante:** contato entre mãos e mão na frente do rosto são casos difíceis ([Moryossef 2021](literatura/moryossef-2021-pose-estimation.md)); a perda de mãos durante o sinal vai de 5% a 19% conforme a fonte (seção 6.3).
- **Licenças:** o EQT-UECE é CC BY-NC 4.0 (sem uso comercial).

---

## 15. A pesquisa em que nos baseamos

Os fichamentos completos estão em [docs/literatura/](literatura/README-indice.md); o capítulo de trabalhos relacionados, em [relatório §2](relatorio/02-trabalhos-relacionados.md).

| Escolha do projeto | Trabalhos que a fundamentam |
|---|---|
| Landmarks MediaPipe em vez de pixels | [Bohacek 2022 (SPOTER + MediaPipe)](literatura/bohacek-2022-spoter-mediapipe.md), [Li 2025 (Uni-Sign)](literatura/li-2025-uni-sign.md), [Sohn 2023 (Kaggle ISLR)](literatura/sohn-2023-kaggle-islr.md), [De Coster 2023](literatura/decoster-2023-sign-embeddings.md); contraponto: [Costa 2025](literatura/costa-2025-i3d-libras.md), [Desai 2023 (ASL Citizen)](literatura/desai-2023-asl-citizen.md) |
| Subconjunto de pontos e preenchimento de falhas | [dos Santos 2025](literatura/santos-2025-landmark-subset.md) (Libras) |
| Normalização pelo corpo + mão local | [De Coster 2023](literatura/decoster-2023-sign-embeddings.md), [Zelezny 2025](literatura/zelezny-2025-pose-ablation.md), [Bohacek 2022 (SPOTER)](literatura/bohacek-2022-spoter.md) |
| Velocidade como feature, espelhamento, reamostragem | [Sohn 2023](literatura/sohn-2023-kaggle-islr.md) |
| Avaliação independente de sinalizador (LOSO) | [Alves 2024](literatura/alves-2024-skeleton-image.md), [dos Santos 2025](literatura/santos-2025-landmark-subset.md) (Libras); [Desai 2023](literatura/desai-2023-asl-citizen.md) |
| Modelos de grafo e transformers como próximos passos | [Jiang 2021 (SAM-SLR)](literatura/jiang-2021-sam-slr.md), [Selvaraj 2022 (OpenHands)](literatura/selvaraj-2022-openhands.md) |
| Teste fora de domínio no MALTA | [Delucis 2025](literatura/delucis-2025-malta-libras.md) |
| Crescer o vocabulário (protótipos, busca em dicionário, fonologia) | [Vandendriessche 2025](literatura/vandendriessche-2025-one-shot.md), [Kezar 2023](literatura/kezar-2023-phonology.md), [Sem-Lex](literatura/kezar-2023-sem-lex.md); *weight imprinting*: Qi, Brown e Lowe, CVPR 2018 (não fichado) |
| CTC para frases | Graves et al., ICML 2006 (não fichado); uso padrão no benchmark de reconhecimento contínuo RWTH-PHOENIX-Weather (a fichar) |

---

## 16. Glossário

| Termo | Significado |
|---|---|
| **Landmark** | ponto-chave do corpo (ex.: ponta do indicador) com coordenadas em cada quadro |
| **MediaPipe** | biblioteca do Google que detecta landmarks de mãos, corpo e rosto em imagens |
| **Glosa** | rótulo convencional de um sinal, em maiúsculas (AMARELO) |
| **Sinalizador** | pessoa que sinaliza nos vídeos |
| **Feature** | número que descreve a entrada (uma coordenada, uma velocidade...) |
| **Embedding** | vetor que resume um vídeo inteiro; vídeos do mesmo sinal devem ter embeddings próximos |
| **Protótipo** | média dos embeddings de um sinal; serve de vetor de pesos do classificador |
| **GRU** | rede recorrente que lê uma sequência passo a passo mantendo uma memória |
| **Época** | uma passada por todo o conjunto de treino |
| **Aumento de dados** | variações artificiais dos exemplos de treino |
| **LOSO** | *leave-one-signer-out*: testar em cada pessoa com um modelo que nunca a viu |
| **Dobra (fold)** | uma das rodadas do LOSO |
| **Top-1 / Top-5** | acerto no primeiro palpite / entre os cinco primeiros |
| **F1-macro** | média, entre as classes, do equilíbrio entre precisão e revocação |
| **CTC** | função de perda que treina reconhecimento de sequências sem alinhamento por quadro |
| **Blank** | símbolo "nenhum sinal novo" da CTC |
| **WER** | taxa de erro por palavra (aqui, por glosa) em frases |
| **Fora de domínio** | dados de condições diferentes das do treino |
| **Snapshot** | versão congelada do registro de dados, usada por um treino |
| **ADR** | *Architecture Decision Record*: documento curto que registra uma decisão e o porquê |

---

## 17. Onde está cada coisa no repositório

| Caminho | Conteúdo |
|---|---|
| `src/libras/registry/` | registro canônico dos dados (quem, qual sinal, qual licença) |
| `src/libras/extract/` | extração de landmarks com MediaPipe |
| `src/libras/data/` | carregamento, pré-processamento, aumento de dados e divisões LOSO |
| `src/libras/models/` | codificadores (MLP, GRU) e classificador de cosseno |
| `src/libras/train/` | treino e avaliação (`python -m libras.train --config ...`) |
| `configs/extraction/` | configurações da extração (`mp_v2.yaml` é a padrão) |
| `configs/experiments/` | configurações de cada experimento de treino |
| `docs/decisoes/` | ADRs, uma decisão por arquivo |
| `docs/relatorio/` | relatório vivo, um capítulo por arquivo |
| `docs/literatura/` | fichamentos dos artigos |
| `docs/datasets/` | fichas dos datasets |
| `LIBRAS_DATA_ROOT` (fora do git) | vídeos, landmarks, registro gerado, experimentos e `PROVENANCE.jsonl` |
