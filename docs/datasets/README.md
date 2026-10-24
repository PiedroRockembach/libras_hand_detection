# Datasets

Uma ficha por dataset (`nome-do-dataset.md`), seguindo o modelo abaixo. Nada aqui é considerado verificado sem a fonte primária (artigo ou página oficial) citada.

## Candidatos (levantamento da Fase 1, 2026-10-22)

Números verificados na fonte primária citada em cada ficha. "a.v." = a verificar. "Indep. sinalizador": o protocolo oficial (ou o mais usado) separa as pessoas do treino e do teste?

### Libras

| Dataset | Tipo | Sinais | Sinalizadores | Amostras | Indep. sinalizador | Licença / acesso | Ficha |
|---|---|---|---|---|---|---|---|
| MINDS-Libras | Isolados, RGB + RGB-D (Kinect v2) | 20 | 12 | 1.200 nominais (efetivo a.v.) | Sem divisão oficial; LOSO possível e usado | CC BY 4.0, download direto (Zenodo) | [minds-libras.md](minds-libras.md) |
| LIBRAS-UFOP | Isolados, pares mínimos, RGB-D (Kinect v1) | 56 | 5 | 3.040 | Sem divisão no Zenodo; LOPO usado na literatura | CC BY 4.0 declarada, mas arquivos restritos: acordo assinado por e-mail | [libras-ufop.md](libras-ufop.md) |
| V-LIBRASIL | Isolados, RGB Full HD | 1.364 termos | 3 papéis de articulador (≈6 pessoas, a.v.) | 4.089 (1 por articulador) | Possível (2 treino / 1 teste), mas ≤ 2 exemplos por classe | Pesquisa/educação, sem redistribuição; IEEE DataPort (login) | [v-librasil.md](v-librasil.md) |
| MALTA-LIBRAS (PUCRS) | Isolados, fontes públicas variadas | a.v. (20 no subconjunto MINDS) | a.v. | a.v. (132 clipes no teste MINDS) | Teste OOD em relação ao MINDS | MIT declarada no HF; licenças das fontes a.v. | [malta-libras.md](malta-libras.md) |
| LIBRAS-EQT-UECE | Isolados + contínuos, RGB + landmarks | a.v. (48 segundo terceiros) | a.v. (6 segundo terceiros) | a.v. | a.v. | CC BY-NC 4.0 (v4; v1.1 era CC BY), download direto | [libras-eqt-uece.md](libras-eqt-uece.md) |
| Libras-UFPel Corpus | Isolados + sentenças + naturalístico, 60 fps | a.v. (2.400 registros isolados) | 4 surdos nativos (controlado) | 4.800 controlados | — | Ainda não liberado (acesso escalonado previsto) | [libras-ufpel-corpus.md](libras-ufpel-corpus.md) |
| Corpus de Libras (UFSC) | Contínuo/espontâneo, anotado | a.v. | a.v. | a.v. | — | a.v.; cadastro no portal | [corpus-de-libras-ufsc.md](corpus-de-libras-ufsc.md) |
| Signbank da Libras (UFSC) | Léxico/dicionário | 3.118 (fonte secundária, a.v.) | a.v. | a.v. | — | a.v. | [signbank-libras-ufsc.md](signbank-libras-ufsc.md) |
| Dicionário INES (v3) | Dicionário em vídeo | a.v. | a.v. | a.v. | — | a.v.; sem download; exige autorização | [dicionario-ines.md](dicionario-ines.md) |
| Alfabeto (Lima et al., 2019, UFPB) | Alfabeto, quadros de vídeo | Letras do alfabeto | 12 | 224.000 imagens | Sim (relatado: 71% indep. vs 99% dep.) | a.v.; contato com autores | [alfabeto-libras-lima-2019.md](alfabeto-libras-lima-2019.md) |
| Alfabeto (Passos et al., 2020) | Imagens estáticas (derivadas de ASL Kaggle) | 15 letras estáticas | a.v. | 4.411 | Não (sem IDs) | CC BY 4.0 (origem a.v.) | [alfabeto-libras-passos-2020.md](alfabeto-libras-passos-2020.md) |
| LSWH100 | Configurações de mão, sintético | 100 configurações | 0 (sintético) | 480 RGB por classe | Não se aplica | Artigo CC BY-NC 4.0; dados a.v. (Zenodo) | [lswh100.md](lswh100.md) |
| Libras Movement (UCI) | Trajetória do centróide, sem vídeo | 15 movimentos | a.v. | 360 | Não | CC BY 4.0 | [libras-movement-uci.md](libras-movement-uci.md) |

### Outras línguas (pré-treino / comparação)

| Dataset | Língua | Sinais | Sinalizadores | Amostras | Indep. sinalizador | Landmarks prontos | Licença | Ficha |
|---|---|---|---|---|---|---|---|---|
| ASL Citizen | ASL | 2.731 | 52 | 83.399 | Sim (35/6/11) | Não (baseline usa MediaPipe) | MSR License: pesquisa não comercial, sem redistribuição | [asl-citizen.md](asl-citizen.md) |
| PopSign ASL v1.0 | ASL (uma mão) | 250 | 47 | 214.326 | Sim (31/8/8) | a.v. | CC BY 4.0 | [popsign-asl.md](popsign-asl.md) |
| Google ISLR "ASL Signs" (Kaggle) | ASL | a.v. | a.v. | a.v. | a.v. | Sim (MediaPipe Holistic, segundo fontes secundárias) | a.v. (regras Kaggle) | [google-asl-signs-kaggle.md](google-asl-signs-kaggle.md) |
| AUTSL | Turca | 226 | 43 | 38.336 | Sim (teste com 6 sinalizadores) | Não (esqueleto Kinect) | Só acadêmico | [autsl.md](autsl.md) |
| WLASL | ASL | 2.000 | 119 | 21.083 | **Não** (4:1:1 por glosa) | Não | C-UDA, não comercial | [wlasl.md](wlasl.md) |
| MS-ASL | ASL | 1.000 | 222 | 25.513 | Sim | Não | a.v. | [ms-asl.md](ms-asl.md) |
| LSA64 | Argentina | 64 | 10 | 3.200 | Sem protocolo oficial | Não (posições de mão/cabeça) | CC BY-NC-SA 4.0 | [lsa64.md](lsa64.md) |

Outros recursos mencionados nas fichas, sem ficha própria: CEFET/RJ-Libras (24 sinais, 20 pessoas, 547 vídeos, segundo [SBrT 2021](https://biblioteca.sbrt.org.br/articlefile/2860.pdf), fonte secundária) e Libras91 (configurações de mão, ver [lswh100.md](lswh100.md)).

## Modelo de ficha

```markdown
# <Nome do dataset>

- **Referência:** citação completa + DOI/URL
- **Licença / termos de uso:**
- **Como obter:** (link, formulário, contato)
- **Conteúdo:** nº de sinais, nº de sinalizadores, repetições, total de vídeos, resolução, fps, modalidades
- **Protocolo oficial de avaliação:** existe? é independente de sinalizador?
- **Resultados de referência na literatura:**
- **Status no projeto:** candidato | adotado | descartado (+ ADR)
- **Observações:** (qualidade, viés, problemas encontrados)
```
