# Phonological Perception of Sign Language Models

- **Referência:** Kayo Yin, Jessica Carter, Alex Xijie Lu, Annemarie Kocab. Aceito no CogSci 2026 (segundo o arXiv). arXiv 2606.28667 (27/06/2026). https://arxiv.org/abs/2606.28667
- **Código:** https://github.com/kayoyin/sign-phonology
- **Problema:** Saber se modelos de reconhecimento de sinais aprendem traços fonológicos abstratos ou apenas padrões estatísticos de baixo nível.
- **Método:**
  - Modelos comparados:
    - **pixel**: I3D com inicialização aleatória, pré-treinado no Kinetics ou treinado no ASL Citizen;
    - **pose**: ST-GCN com inicialização aleatória ou treinado no ASL Citizen.
  - Os modelos treinados no ASL Citizen usam uma redivisão que exclui os sinais de teste.
  - Mede-se a sensibilidade das representações a **pares mínimos** (diferença em um só parâmetro) e a correlação com julgamentos humanos de similaridade.
- **Dados e protocolo:**
  - 259 pares mínimos do ASL Citizen: 137 de configuração de mão, 31 de local, 73 de movimento e 18 de orientação/não-manual.
  - Avaliação fora do domínio com o Sem-Lex (N = 71).
  - Estímulos controlados de configuração de mão (HCS) com julgamentos humanos.
- **Resultados principais** (% de pares em que o modelo é significativamente mais sensível que o baseline aleatório, ASL Citizen; fonte: arXiv 2606.28667):

  | Parâmetro | I3D-ASL | STGCN-ASL |
  |---|---|---|
  | Total | 69,50 | 81,08 |
  | Configuração de mão | 69,34 | **86,13** |
  | Local | **80,65** | 67,74 |
  | Movimento | 73,97 | 80,82 |

  - No Sem-Lex (fora do domínio) a sensibilidade cai para 45,07 (I3D-ASL) e 49,30 (STGCN-ASL).
  - Correlação com a similaridade percebida por sinalizadores: STGCN-ASL r = 0,49; I3D-ASL 0,31; I3D-Kinetics 0,19.
- **Relevância para este projeto:**
  - Comparação direta pose × pixel em nível **fonológico**, não só de acurácia.
  - A pose capta melhor **configuração de mão e movimento**; o pixel capta melhor **local** (ponto de articulação).
  - Para a Libras, isso indica que o ponto de articulação é o calcanhar de Aquiles dos landmarks. Ele depende de pontos do corpo e da face, e o 1º lugar do Kaggle descartou o corpo.
  - Justifica a ablação mãos / mãos+pose / mãos+pose+face prevista no ADR-0002 e *features* relativas mão–rosto e mão–corpo.
- **Limitações / críticas:**
  - Só ASL, só dois modelos (I3D e ST-GCN).
  - Poucos pares de local (31).
  - As representações são frágeis sob mudança de domínio.
