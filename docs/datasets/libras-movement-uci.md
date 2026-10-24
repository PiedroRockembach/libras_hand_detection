# Libras Movement (UCI)

- **Referência:** Dias, D.; Peres, S.; Bíscaro, H. *Libras Movement* [Dataset]. UCI Machine Learning Repository, 2009. DOI [10.24432/C5GC82](https://doi.org/10.24432/C5GC82) ([página UCI](https://archive.ics.uci.edu/dataset/181/libras+movement)).
- **Licença / termos de uso:** **CC BY 4.0** ([UCI](https://archive.ics.uci.edu/dataset/181/libras+movement)).
- **Como obter:** download direto no UCI.
- **Conteúdo:** 360 instâncias, 15 classes × 24 instâncias. Cada classe é "um tipo de movimento de mão em Libras". Cada vídeo foi normalizado para 45 quadros; em cada quadro foi tomado o centróide da mão segmentada, gerando 90 atributos numéricos (45 pontos × 2 coordenadas) ([UCI](https://archive.ics.uci.edu/dataset/181/libras+movement)). **Não há vídeos**, só a trajetória do centróide. Número de sinalizadores: não informado (**a verificar**).
- **Protocolo oficial de avaliação:** não há divisão por sinalizador (não há ID de sinalizador).
- **Resultados de referência na literatura:** não levantados (o dataset é usado como benchmark genérico de classificação e agrupamento).
- **Status no projeto:** candidato. **Recomendação: descartar** (a decisão cabe à sessão principal, via ADR se aplicável).
- **Observações:** representa só o parâmetro movimento, numa trajetória 2D de um ponto, que é a limitação que o ADR-0001 critica no fork antigo. Não tem vídeo nem landmarks de mão e não permite avaliação independente de sinalizador. Pode no máximo servir de referência histórica.
