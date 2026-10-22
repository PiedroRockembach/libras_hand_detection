# Google - Isolated Sign Language Recognition ("ASL Signs", Kaggle)

- **Referência:** competição Kaggle *Google - Isolated Sign Language Recognition*, 2023 ([kaggle.com/competitions/asl-signs](https://www.kaggle.com/competitions/asl-signs/overview)). Artigo de dados relacionado (mesmo projeto, PopSign): Starner et al., NeurIPS 2023 D&B ([PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/00dada608b8db212ea7d9d92b24c68de-Paper-Datasets_and_Benchmarks.pdf)).
- **Licença / termos de uso:** **a verificar.** As páginas Data e Rules do Kaggle exigem JavaScript/login e não puderam ser lidas ([rules](https://www.kaggle.com/competitions/asl-signs/rules)). Até ser confirmada, tratar como uso restrito às regras da competição.
- **Como obter:** conta Kaggle e aceite das regras da competição (**a verificar**).
- **Conteúdo:** **a verificar** em fonte primária. Fontes secundárias descrevem landmarks **MediaPipe Holistic** por quadro (mãos, pose e face), sem vídeo, com o objetivo de melhorar o jogo PopSign ([hackster.io](https://hackster.io/AlbertaBeef/insightful-datasets-for-asl-recognition-f786b9); [GitHub de participante](https://github.com/JosephZahar/Google-Isolated-Sign-Language-Recognition-Kaggle)). Número de sinais, sequências e participantes: **a verificar** (não cito os números que circulam por não terem fonte primária lida).
- **Protocolo oficial de avaliação:** teste oculto da competição. Se é independente de participante: **a verificar.**
- **Resultados de referência na literatura:** placar da competição: **a verificar.**
- **Status no projeto:** candidato (pré-treino com landmarks prontos, se a licença permitir).
- **Observações:**
  - É o único grande dataset com landmarks Holistic **já extraídos**, o mesmo formato do ADR-0002, o que evita extração.
  - A versão do MediaPipe e o pré-processamento podem diferir dos nossos. Misturar landmarks de extratores diferentes é uma fonte de viés a controlar.
  - Pendência: baixar com conta Kaggle e registrar aqui a licença e os números do `train.csv`.
