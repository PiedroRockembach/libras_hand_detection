# Signbank da Libras (UFSC)

- **Referência:** Pizzio, A. L. et al. "SignBank da Libras". Fórum Linguístico, dez. 2020, DOI [10.5007/1984-8412.2020.e77342](https://doi.org/10.5007/1984-8412.2020.e77342) ([página do artigo](https://periodicos.ufsc.br/index.php/forum/article/view/77342)). Portal: [signbank.libras.ufsc.br](https://signbank.libras.ufsc.br/), desenvolvido pelo Levante Lab.
- **Licença / termos de uso:** **a verificar.** A página "Sobre" não carregou na consulta de 2026-10-22 ("Erro ao carregar as informações") ([Sobre](https://signbank.libras.ufsc.br/pt/about)).
- **Como obter:** consulta web por sinal ([signbank.libras.ufsc.br](https://signbank.libras.ufsc.br/)). Download em lote ou API: **a verificar.**
- **Conteúdo:** banco lexical baseado no modelo dos Signbanks internacionais, com aspectos fonológicos da Libras, que importou os dados do antigo "Identificador de Sinais" (segundo o artigo de 2020, conforme resumo indexado; **a verificar** no texto completo). Tamanho: **3.118 sinais** segundo o compêndio de recursos da Univ. de Hamburgo (fonte secundária; [sign-lang.uni-hamburg.de](https://www.sign-lang.uni-hamburg.de/lr/compendium/lex/signbanklibras.html), que estava inacessível na consulta). Número de sinalizadores, vídeos por sinal, resolução e fps: **a verificar.**
- **Protocolo oficial de avaliação:** não existe (é um dicionário/léxico).
- **Resultados de referência na literatura:** nenhum.
- **Status no projeto:** candidato (fonte de vocabulário e de **anotação fonológica**).
- **Observações:**
  - Interesse principal: se cada entrada traz configuração de mão, locação e movimento anotados, o Signbank vira um **rótulo fonológico por sinal**. Isso é o que a ideia de decompor o reconhecimento em parâmetros (ADR-0001) precisa para escalar o vocabulário. Confirmar quais campos estão disponíveis.
  - Provavelmente tem 1 vídeo por sinal. Serve para *zero/few-shot* e para definir o vocabulário-alvo, não para treino direto.
  - **2026-10-24:** escolhido como base do vocabulário-alvo, por pedido ao grupo, sem raspagem (ADR-0013). O portal busca os dados numa API separada (`api-signbank.levantelab.com.br`), não tem `robots.txt` nem termos de uso publicados, e a política de privacidade publicada é texto provisório. O painel administrativo referencia vídeos com anotação ELAN (`.eaf`); se elas são públicas e trazem parâmetros fonológicos: **a verificar**.
