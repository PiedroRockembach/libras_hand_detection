# LSA64 (Língua de Sinais Argentina)

- **Referência:** Ronchetti, F.; Quiroga, F.; Estrebou, C.; Lanzarini, L.; Rosete, A. *LSA64: A Dataset of Argentinian Sign Language*. XXII CACIC, 2016. Página: [facundoq.github.io/datasets/lsa64](https://facundoq.github.io/datasets/lsa64/).
- **Licença / termos de uso:** **CC BY-NC-SA 4.0**: uso acadêmico, educacional e pessoal; sem uso comercial; derivados sob a mesma licença ([página](https://facundoq.github.io/datasets/lsa64/)).
- **Como obter:** links para Mega e Google Drive: bruto (1,9 GB), cortado (1,5 GB) e pré-processado (800 MB) ([página](https://facundoq.github.io/datasets/lsa64/)).
- **Conteúdo:** 64 sinais (42 com uma mão, 22 com duas), **10 sinalizadores** não fluentes, todos destros, 5 repetições, **3.200 vídeos**. 1920×1080 a **60 fps**. Os sinalizadores usaram **luvas fluorescentes** e roupa preta, em frente a parede branca. A sessão 1 foi externa e a sessão 2 interna. O sinalizador 10 da sessão 1 foi substituído na sessão 2 ([página](https://facundoq.github.io/datasets/lsa64/)). A versão pré-processada traz posições de mão e cabeça por quadro, não landmarks MediaPipe.
- **Protocolo oficial de avaliação:** não especificado ([página](https://facundoq.github.io/datasets/lsa64/)). É possível fazer LOSO.
- **Resultados de referência na literatura:** não levantados nesta ficha. Muitos trabalhos usam divisão aleatória; verificar o protocolo de cada número antes de comparar.
- **Status no projeto:** candidato (comparação secundária).
- **Observações:** as luvas coloridas criam uma distribuição visual atípica. Para landmarks o efeito deve ser menor, mas é preciso medir a detecção do MediaPipe com luvas. A troca do sinalizador 10 entre sessões exige cuidado com o ID no LOSO.
