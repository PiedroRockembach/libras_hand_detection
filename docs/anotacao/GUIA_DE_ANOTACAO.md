# Guia de anotação: revisão de glosas e frases em Libras

Obrigado por ajudar! Este projeto de pesquisa treina um modelo de computador para reconhecer sinais da Libras em vídeo. O objetivo é chegar a um tradutor que entenda **frases**, e não só sinais soltos. Para isso, o modelo precisa saber **quais sinais aparecem em cada frase e em que ordem**. É essa informação que pedimos a você.

São duas tarefas. Faça a Tarefa 1 primeiro, porque a Tarefa 2 reaproveita as glosas revisadas nela.

## Antes de começar

- Os arquivos `.csv` abrem no **LibreOffice Calc** ou no **Excel**. Se o programa perguntar, escolha a codificação **UTF-8** e o separador **vírgula**.
- Ao salvar, mantenha o formato **CSV** (no LibreOffice, "Manter formato atual").
- **Não altere** as colunas que já vêm preenchidas (`gloss_id`, `glosa_atual` etc.). Escreva só nas colunas vazias.
- Os vídeos estão na pasta `clips/`. O nome de cada arquivo começa pelo código da linha na planilha. Por exemplo, `G00021_A_ENERGIA_CAIU_NA_CASA_TODA_1.mp4` é um exemplo da linha `G00021`. Cada exemplo mostra uma pessoa diferente.
- Na dúvida, escreva a dúvida na coluna `observacoes` e siga em frente.

## Tarefa 1: revisão das glosas (`revisao_glosas.csv`, 195 linhas)

Cada linha é um sinal ou uma frase dos datasets que usamos. Os nomes vieram dos arquivos originais e podem ter erros.

| Coluna | O que fazer |
|---|---|
| `glosa_atual`, `tipo_atual` | Como está hoje (só leitura) |
| `rotulos_originais` | Nome que o sinal tinha no dataset de origem (só leitura) |
| `video_1`, `video_2` | Vídeos de exemplo |
| `aviso` | Quando preenchido, leia com atenção. Por exemplo, há vídeos idênticos rotulados com duas frases diferentes, e queremos saber qual rótulo está certo |
| `glosa_corrigida` | Preencha **só se** a glosa atual estiver errada ou mal escrita (ex.: `MACA` → `MAÇÃ`) |
| `tipo_corrigido` | Preencha **só se** o tipo estiver errado. `sinal` = um único sinal; `frase` = mais de um sinal. Exemplo: `NÃO SEI` está como frase, mas pode ser um sinal único |
| `variante_regional` | Se o sinal do vídeo for uma variante regional, indique qual (se souber) |
| `observacoes` | Qualquer coisa útil, por exemplo "BANCO aqui é banco de dinheiro". **Se a linha estiver toda correta, escreva `OK`.** Assim sabemos que ela foi conferida, e não esquecida |

## Tarefa 2: sequência de sinais das frases (`anotacao_frases.csv`, 131 linhas)

Cada linha é uma frase em português que foi sinalizada por várias pessoas. Assista aos vídeos e escreva, na coluna `sequencia_glosas_libras`, **os sinais que aparecem, na ordem em que aparecem**.

Exemplo (ilustrativo, não é a resposta):

| frase_portugues | sequencia_glosas_libras |
|---|---|
| Um poste caiu. | POSTE CAIR |
| Eu quero pagar com o Pix. | EU QUERER PAGAR P-I-X |

### Convenções

Estas convenções são uma proposta. Se você usa outra que acha melhor, combine com o pesquisador **antes** de começar e use a mesma em todas as linhas.

1. **Um sinal = uma glosa**, em MAIÚSCULAS, separadas por espaço.
2. **Glosa na forma neutra:** verbo no infinitivo (`CAIR`, não `CAIU`), sem flexão de gênero ou número quando o sinal não muda.
3. **Só o que é sinalizado.** Artigos, preposições e outras palavras do português que não têm sinal não entram.
4. **Reaproveite as glosas da Tarefa 1.** Se o sinal for o mesmo de uma glosa que já existe (por exemplo, `ENERGIA`, `CAIR`, `POSTE`), use exatamente a mesma escrita. É isso que permite ao modelo ligar o sinal isolado ao sinal dentro da frase.
5. **Sinal único traduzido por várias palavras:** junte com hífen (`NÃO-SABER`).
6. **Datilologia** (soletração com o alfabeto manual): letras separadas por hífen (`P-I-X`, `C-P-F`).
7. **Números:** por extenso (`SETE`, `QUINZE`).
8. **Apontação:**
   - para pronomes pessoais, use o pronome (`EU`, `VOCÊ`, `ELE`);
   - para outras apontações, use `IX(referente)`, por exemplo `IX(LÁ)`.
9. **Repetição** que muda o sentido (plural, intensidade): `+` no fim (`CASA+`).
10. **Classificador:** `CL(descrição curta)`, por exemplo `CL(poste-caindo)`.
11. **Expressão facial ou corporal com função gramatical** (pergunta, negação, intensidade): anote em `observacoes`, por exemplo "negação com a cabeça em CONTA PAGO".
12. **Pessoas sinalizam de jeitos diferentes:** escreva a sequência que aparece na maioria dos vídeos e descreva as diferenças em `observacoes`, citando o vídeo.
13. **Trecho que você não reconhece:** escreva `?` no lugar do sinal e explique em `observacoes`.

## Ao terminar

Salve os dois arquivos em CSV e avise o pesquisador. Não renomeie os arquivos nem mova a pasta `clips/`. As respostas serão importadas para a base de dados do projeto, e cada correção fica registrada com a data e o motivo.
