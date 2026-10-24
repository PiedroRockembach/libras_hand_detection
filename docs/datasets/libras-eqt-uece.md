# LIBRAS-EQT-UECE

- **Referência:** Moraes, J.; Moreira, E.; Uchoa, I.; Nascimento, A.; Ferreira, I.; Dias, C.; Chagas, M.; Santos, I.; Cavalcante, M.; Maia, P.; Santos, I.; Araujo, T. *LIBRAS-EQT-UECE: A Dataset for Isolated and Continuous Brazilian Sign Language Recognition and Translation*. Zenodo, v4, 2026-07-16, DOI [10.5281/zenodo.21398895](https://zenodo.org/records/21398895). Versão anterior: v1.1, 2026-06-01, [10.5281/zenodo.20497742](https://zenodo.org/records/20497742). Afiliação: Universidade Estadual do Ceará (UECE) e Grupo Equatorial. Artigo associado: **a verificar** (a seção "related publications" do Zenodo está vazia).
- **Licença / termos de uso:** **CC BY-NC 4.0** na v4, que proíbe uso comercial ([Zenodo v4](https://zenodo.org/records/21398895); [LICENSE.txt](https://zenodo.org/records/21398895/files/LICENSE.txt?download=1)). Atenção: a **v1.1 estava sob CC BY 4.0** ([Zenodo v1.1](https://zenodo.org/records/20497742)). A licença mudou entre versões; registrar qual versão for baixada.
- **Como obter:** download aberto no Zenodo (45,4 GB): `Landmarks.zip` (773,6 MB), `Libras-EQT-UECE (RGB).zip` (9,0 GB) e `Libras-EQT-UECE-T.zip` (35,6 GB, formato estilo PHOENIX com anotações `.npy`, quadros 256×256 e features) ([Zenodo](https://zenodo.org/records/21398895); [README](https://zenodo.org/records/21398895/files/README.md?download=1)).
- **Conteúdo:**
  - **Sinais:** amostras isoladas **e** contínuas, organizadas por glosa e informante ([README](https://zenodo.org/records/21398895/files/README.md?download=1)). Número de classes isoladas: uma biblioteca de terceiros filtra e padroniza **48** ([PyPI datasets-islr-libras](https://pypi.org/pypi/datasets-islr-libras/json)). O número oficial está **a verificar** no README/artigo.
  - **Sinalizadores:** 6 informantes, segundo a mesma biblioteca de terceiros ([PyPI](https://pypi.org/pypi/datasets-islr-libras/json)). **A verificar** na fonte primária.
  - **Repetições:** 6 por informante, segundo a biblioteca de terceiros ([PyPI](https://pypi.org/pypi/datasets-islr-libras/json)). **A verificar.**
  - **Total / resolução / fps:** **a verificar.**
  - **Landmarks pré-computados:** **sim** (`Landmarks.zip`, CSV ou JSON) ([README](https://zenodo.org/records/21398895/files/README.md?download=1)). O extrator (MediaPipe? qual versão?) está **a verificar.**
- **Protocolo oficial de avaliação:** o pacote "-T" traz divisões train/dev/test ([README](https://zenodo.org/records/21398895/files/README.md?download=1)); se são independentes de sinalizador está **a verificar**. A biblioteca de terceiros propõe 5 dobras estratificadas por sinalizador (3 treino / 1 validação / 1 teste) ([PyPI](https://pypi.org/pypi/datasets-islr-libras/json)). Não é oficial.
- **Resultados de referência na literatura:** **a verificar** (nenhuma publicação localizada).
- **Status no projeto:** candidato.
- **Observações:**
  - É recente (2026), aberto e já traz landmarks, além de cobrir isolado e contínuo, o que serve à Fase 6.
  - Quase todos os números vêm por enquanto de uma biblioteca de terceiros. Antes de adotar, baixar o README completo e o `Landmarks.zip` e confirmar classes, informantes e extrator.
  - A licença NC é compatível com pesquisa acadêmica e publicação, mas não com produto comercial.
