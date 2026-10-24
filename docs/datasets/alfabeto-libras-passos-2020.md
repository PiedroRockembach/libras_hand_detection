# Brazilian Sign Language Alphabet (Passos et al., 2020)

- **Referência:** Passos, B. T.; Fernandes, A. M. R.; Comunello, E. *Brazilian Sign Language Alphabet*. Mendeley Data, 2020. DOI v3: [10.17632/k4gs3bmx5k.3](https://data.mendeley.com/datasets/k4gs3bmx5k/3); v5: [10.17632/k4gs3bmx5k.5](https://data.mendeley.com/datasets/k4gs3bmx5k/5). Página: [biankatpas.github.io/Brazilian-Sign-Language-Alphabet-Dataset](https://biankatpas.github.io/Brazilian-Sign-Language-Alphabet-Dataset/). Instituição: UNIVALI.
- **Licença / termos de uso:** **CC BY 4.0** na v3 do Mendeley ([Mendeley v3](https://data.mendeley.com/datasets/k4gs3bmx5k/3)). Atenção: as imagens são **derivadas do Kaggle ASL Alphabet** (grassknoted/asl-alphabet) ([Mendeley v3](https://data.mendeley.com/datasets/k4gs3bmx5k/3)). A licença da fonte original está **a verificar**, e pode não permitir a relicença em CC BY.
- **Como obter:** download direto no Mendeley Data ou no GitHub dos autores.
- **Conteúdo:**
  - **Classes:** 15 letras **estáticas** (A, B, C, D, E, I, L, M, N, O, R, S, U, V, W), escolhidas por terem configuração igual entre ASL e Libras ([Mendeley v3](https://data.mendeley.com/datasets/k4gs3bmx5k/3)).
  - **Total:** 4.411 imagens de 200×200, com 150 a 600 por classe e anotação de *bounding box* (LabelImg/XML). A descrição também diz "3.000 imagens de cada sinal", o que é **inconsistente** com o total ([Mendeley v3](https://data.mendeley.com/datasets/k4gs3bmx5k/3)).
  - **Sinalizadores:** não informados (**a verificar**).
  - **Modalidade:** imagens estáticas, não vídeo. Não há landmarks.
- **Protocolo oficial de avaliação:** não existe e não há IDs de sinalizador. Sem eles, não dá para fazer avaliação independente de sinalizador.
- **Resultados de referência na literatura:** não levantados.
- **Status no projeto:** candidato. **Recomendação: não adotar como dataset principal.**
- **Observações:** não é Libras gravada como Libras: são letras de ASL que coincidem visualmente com Libras. Não tem letras dinâmicas (H, J, K, X, Y, Z, Ç). Serve no máximo como teste rápido de configuração de mão estática. O mais citado do tipo encontrado, mas de qualidade baixa para este projeto.
