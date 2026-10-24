# LSWH100 (configurações de mão da Libras, sintético)

- **Referência:** Lobo-Neto, V. C.; Pedrini, H. *LSWH100: A handshape dataset for Brazilian sign language (Libras) using SignWriting*. Data in Brief, 56:110780, 2024. DOI [10.1016/j.dib.2024.110780](https://doi.org/10.1016/j.dib.2024.110780) (metadados via [Crossref](https://api.crossref.org/works?query.bibliographic=LSWH100+handshape+dataset+Libras+SignWriting&rows=2); texto em [PMC11357844](https://pmc.ncbi.nlm.nih.gov/articles/PMC11357844/)). Dados: [Zenodo 10.5281/zenodo.10635628](https://zenodo.org/records/10635628).
- **Licença / termos de uso:** o artigo está sob **CC BY-NC 4.0** ([Crossref](https://api.crossref.org/works?query.bibliographic=LSWH100+handshape+dataset+Libras+SignWriting&rows=2); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11357844/)). A licença do registro Zenodo dos dados está **a verificar** (a tabela de especificações não a informa).
- **Como obter:** Zenodo ([10.5281/zenodo.10635628](https://zenodo.org/records/10635628)).
- **Conteúdo:**
  - **Classes:** 100 configurações de mão da Libras, nomeadas pela convenção SignWriting ([Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1016/j.dib.2024.110780&resultType=core&format=json)).
  - **Dados:** **sintéticos**, renderizados no Blender a partir de malha 3D de mão realista. **Sem sinalizadores humanos** ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11357844/)).
  - **Por classe:** 480 imagens RGB 500×500 (400 treino / 40 validação / 40 teste), em 4 perspectivas, cada uma com mapa de profundidade e máscara. O resumo fala em 144.000 imagens no total ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11357844/); [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1016/j.dib.2024.110780&resultType=core&format=json)).
  - **Anotações:** COCO com classe, *bounding box*, máscara de segmentação e **21 keypoints da mão** ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11357844/)). Os 21 pontos coincidem em número com o MediaPipe Hands; a compatibilidade de ordem e topologia está **a verificar**.
- **Protocolo oficial de avaliação:** divisão fixa treino/validação/teste por classe. Sinalizador não se aplica.
- **Resultados de referência na literatura:** o artigo de dados não traz *baseline* ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11357844/)).
- **Status no projeto:** candidato (auxiliar: pré-treino ou supervisão do parâmetro configuração de mão).
- **Observações:**
  - É relevante para a direção de **decomposição fonológica** (ADR-0001): oferece rótulos de configuração de mão para 100 classes. Usado sozinho não reconhece sinais, e a lacuna entre dados sintéticos e reais precisa ser medida.
  - Alternativa real: **Libras91** (91 configurações de mão, 108.896 amostras), de Carvalho, Brandão & Ferreira, WVC 2021, DOI [10.5753/wvc.2021.18883](https://doi.org/10.5753/wvc.2021.18883). Disponibilidade, licença e número de sinalizadores estão **a verificar** (repositório de código citado: github.com/gabrielpeixoto-cvai/handarch).
