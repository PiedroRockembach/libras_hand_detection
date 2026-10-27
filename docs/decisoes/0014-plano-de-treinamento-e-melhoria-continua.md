# ADR-0014: Plano de treinamento: sinais isolados, sequências de sinais e melhoria contínua

- **Status:** Aceita (aprovada pelo pesquisador em 2026-10-27)
- **Data:** 2026-10-27
- **Relacionadas:** ADR-0002, ADR-0004, ADR-0009, ADR-0010, ADR-0012, ADR-0013; inventário em [relatório §3.2.1](../relatorio/03-metodologia.md)

## Contexto

Requisitos do modelo:

1. ser treinável **agora**, com os dados disponíveis (§3.2.1);
2. **melhorar com vídeos novos**: sinais novos, pessoas novas, fontes novas (ADR-0009);
3. **requisito crítico** (pesquisador, 2026-10-27): em alguma etapa, reconhecer **sinais em sequência**, como numa frase, identificando onde um sinal termina e o próximo começa. É o que permite chegar a um tradutor real.

O que os dados atuais permitem:

- **MINDS-Libras:** 20 sinais isolados × 12 sinalizadores, rótulos limpos. É a base do baseline e da demo v0.
- **LIBRAS-EQT-UECE:** 44 sinais isolados + **131 frases** (~30 vídeos cada, 5 sinalizadores), no domínio de atendimento de uma distribuidora de energia. Inspeção do pacote `-T` (2026-10-27):
  - o campo `label_gloss` **não contém glosas de Libras**, e sim a frase em português dividida em palavras ("Eu quero pagar as minhas faturas em aberto");
  - não há alinhamento temporal;
  - os sinais isolados parecem ser componentes das frases: 102 das 131 frases contêm pelo menos uma palavra que é sinal isolado no dataset, numa estimativa por palavra do português a confirmar.
- **MALTA-LIBRAS:** teste fora de domínio para sinais isolados (20 do MINDS, 36 do EQT).
- **Nenhum dataset baixado tem frases anotadas com sequência de glosas.** O Corpus de Libras (UFSC) tem material contínuo anotado em ELAN, mas exige cadastro (ficha `corpus-de-libras-ufsc.md`).

## Alternativas consideradas

**A. Como o modelo incorpora dados novos:**

1. **Classificador fixo, retreinado do zero a cada versão:** reproduzível, mas sinais com poucos exemplos ficam mal servidos.
2. **Aprendizado contínuo incremental:** esquece o que aprendeu e depende da ordem dos dados. Não é replicável (regra 4).
3. **Codificador de *embeddings* + classificador de cosseno com pesos inicializados por protótipos (*weight imprinting*), com retreino completo por *snapshot*:** um sinal novo entra sem retreino (protótipo = média dos *embeddings*). O retreino oficial continua reproduzível.

**B. Como reconhecer sequências de sinais:**

1. **Janela deslizante + classificador isolado:** não precisa de dados contínuos, mas não aprende as transições entre sinais (coarticulação) e tende a errar nas fronteiras.
2. **Reconhecimento contínuo com CTC sobre glosas:** o padrão da literatura (PHOENIX). Aprende as fronteiras implicitamente e não precisa de alinhamento por quadro, só da sequência de glosas de cada vídeo. Precisa de frases anotadas com glosas.
3. **Tradução direta vídeo → português, sem glosas:** é o objetivo final de um tradutor, mas exige muito mais dados de frases do que existem em Libras hoje.

## Decisão

**A3 + B2, preparados desde o primeiro modelo.** A1 entra como baseline de comparação. B1 é o fallback da demo se o contínuo não ficar pronto. B3 fica como etapa futura sobre a saída de glosas.

### 1. Arquitetura única, preparada para sequências desde o início

```
landmarks (T × K × C)
  → codificador temporal por quadro (a saída é uma sequência de features, não um vetor único)
      ├─ sinal isolado: pooling temporal → embedding → cosseno/protótipos
      └─ sequência: cosseno por quadro sobre o vocabulário + "blank" → CTC → sequência de glosas
  → (etapa futura) glosas → frase em português (B3 / modelo de linguagem)
```

- O codificador é sempre **por quadro** (convolução temporal, GRU ou Transformer). Isso permite que o mesmo modelo treinado em sinais isolados seja ajustado com CTC para frases, sem trocar de arquitetura.
- A mesma matriz de pesos de cosseno, inicializada pelos protótipos, serve à cabeça isolada e à CTC. Um sinal novo adicionado por protótipo fica disponível nas duas.
- **Tempo real:** a versão da demo usa codificador **causal ou com pouco *lookahead*** (por exemplo, atenção com janela limitada). A latência é medida desde o início.

### 2. Dados para sequências (em ordem de custo)

1. **Anotar as 131 frases do EQT com a sequência de glosas em Libras**, feito por uma pessoa fluente. Cada classe é a mesma frase em ~30 vídeos, então **131 anotações rotulam ~3.900 vídeos de sinalização contínua real**, de 5 sinalizadores. É a ação de maior retorno. As anotações vão para `metadata/registry/` (nova tabela `phrase_glosses`, ADR-0010).
2. **Frases sintéticas:** concatenar landmarks de sinais isolados (MINDS + EQT) com transições interpoladas. É barato e ilimitado, ensina fronteiras, mas não tem coarticulação real. Serve como aumento de dados, nunca como avaliação.
3. **Corpus de Libras (UFSC):** pedir acesso. É material espontâneo e anotado, o teste mais realista.
4. **Parceiros:** o formato de submissão (mês 4) passa a incluir **frases gravadas com sequência de glosas** e um protocolo de anotação (ELAN). O relatório de lacunas mede também a cobertura de frases.

### 3. Pipeline de dados para o treino

```
snapshot do registro (vX.Y)
  → filtros: excluir label_conflict; uma cópia por duplicate_video; licença por finalidade
  → landmarks em cache (ADR-0012), só para amostras novas → controle de qualidade (limiar do piloto)
  → transformações no carregamento (YAML): pontos (mãos + tronco superior + face parcial),
    normalização pelos ombros, reamostragem (isolados: T fixo inicial 64; frases: fps fixo, T variável)
  → aumento de dados: rotação/escala leves, velocidade 0,8–1,2×, descarte de quadros, ruído,
    remoção de uma mão, espelhamento; concatenação sintética (frases)
```

O Kinect do MINDS entra como vista alternativa da mesma amostra, no mesmo grupo de divisão.

### 4. Fases (cronograma do ADR-0009)

| Fase | Mês | Dados | Entregas |
|---|---|---|---|
| A. Baseline isolado | 2 | MINDS (20 sinais) | MLP-estatísticas (sanidade) → codificador por quadro (GRU); LOSO; teste no MALTA; **demo v0** (sinais isolados) |
| B. Diversidade + anotação | 3 | + Kinect como vista, + EQT (44 sinais) | 64 classes; verificação do atalho por dataset. **Em paralelo: anotação de glosas das 131 frases** |
| C. Sequências v0 | 4 | EQT frases (glosas) + frases sintéticas | Ajuste CTC; **WER independente de sinalizador** (LOSO nos 5 do EQT); comparação com a janela deslizante (B1) |
| D. Tempo real e demo v1 | 5 | idem | Codificador causal; detecção de sinais em sequência na webcam, no domínio do EQT (atendimento de energia); glosas → português por regras ou modelo de linguagem, como demonstração |
| E. Além dos 6 meses | — | Corpus UFSC, parceiros | Vocabulário aberto, frases espontâneas, tradução para português |

### 5. Avaliação fixa, comparável entre versões

- **Sinais isolados:** MINDS LOSO (top-1, top-5, F1-macro), MALTA fora de domínio, teste de webcam interno.
- **Sequências:** WER sobre glosas no EQT, com divisão por sinalizador. Precisão de fronteira (início e fim de cada sinal) nas frases sintéticas, onde as fronteiras são conhecidas.
- **Reserva permanente de parceiros:** uma parte dos sinalizadores de cada lote novo vai só para teste.
- Um modelo novo só substitui o anterior sem regressão nos conjuntos fixos. Todo treino registra *snapshot*, configuração, seed e commit (MLflow).

### 6. Simulações de crescimento (fases B–C)

- Curva de sinalizadores (2, 4, 8, 11 no MINDS) e de exemplos por sinal.
- Sinais novos só por protótipo versus retreino completo.
- **Sequências:** WER em função do número de frases anotadas e da proporção de frases sintéticas.

Os resultados alimentam a captação de recursos e o pedido aos parceiros (ADR-0013).

### 7. Ciclo de melhoria com dados novos

```
vídeos novos (sinais isolados e/ou frases com glosas)
  → adaptador / formato de submissão → registry build → snapshot vN+1
  → extração das amostras novas → controle de qualidade → validação humana (rótulos e glosas)
  → (rápido)  protótipos dos sinais novos na demo
  → (oficial) retreino completo (isolado + CTC) no snapshot vN+1, mesma config
  → avaliação nos conjuntos fixos → promoção sem regressão → PROVENANCE.jsonl + MLflow
```

## Justificativa

- Um codificador por quadro desde o primeiro modelo evita uma troca de arquitetura no meio do projeto. O trabalho de sinais isolados vira pré-treino do contínuo.
- O CTC só precisa da sequência de glosas por vídeo, sem alinhamento. Com o desenho do EQT (mesma frase em ~30 vídeos), o custo de anotação cai para 131 frases.
- As frases sintéticas e a janela deslizante dão um caminho de sequência mesmo se a anotação atrasar.
- O domínio do EQT (atendimento de energia) dá à demo de frases um cenário de aplicação concreto para a captação de recursos.

## Consequências

- **Depende de pessoa fluente em Libras** para anotar as 131 frases (fase B). É o principal gargalo humano do plano.
- **Licença:** o EQT-UECE é CC BY-NC 4.0. A demo de frases é de pesquisa; uso com parceiro comercial exige verificação.
- **Mudanças em outros ADRs:** nova tabela `phrase_glosses` no esquema (ADR-0010); o formato de submissão e o relatório de lacunas passam a cobrir frases (ADR-0009, ADR-0013); pedir acesso ao Corpus de Libras UFSC.
- **Pré-requisitos:** piloto de extração (taxa de detecção, limiar); confirmar se Canon e Kinect são a mesma execução; `docker-buildx`.
- **Código novo:** `libras.data` (isolado + frases + concatenação sintética), `libras.models` (codificador por quadro, cosseno, CTC), `libras.train`, `libras.eval` (LOSO, WER, conjuntos fixos).
- **Em aberto, a fixar com experimentos:** T e fps, pontos da face, limiar de rejeição, LOSO simples ou aninhado, dimensão do *embedding*, *lookahead* do modelo causal.
- **Risco:** com 5 sinalizadores e 131 frases de um só domínio, o reconhecimento de frases da POC é de **vocabulário e domínio fechados**. Um tradutor aberto depende de dados de parceiros e do Corpus UFSC (fase E). Isso precisa ficar explícito nas apresentações.
