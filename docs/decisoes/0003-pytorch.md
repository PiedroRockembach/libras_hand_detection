# ADR-0003: PyTorch em vez de TensorFlow

- **Status:** Proposta
- **Data:** 2026-10-22

## Contexto

O fork usava TensorFlow/Keras com exportação TFLite. O treino será numa RTX 5090 (arquitetura Blackwell, sm_120), e o desenvolvimento acontece em Windows.

## Alternativas consideradas

1. **TensorFlow/Keras:** continuidade com o fork. Porém o TensorFlow não oferece GPU no Windows nativo desde a versão 2.10 (exigiria WSL2), e o suporte a Blackwell precisaria ser verificado.
2. **PyTorch:** wheels oficiais com CUDA 12.8 (a partir do 2.7) suportam sm_120 e funcionam no Windows nativo. A maioria das implementações de referência de reconhecimento de sinais por esqueleto (ST-GCN, SL-GCN, transformers) está em PyTorch.

## Decisão

PyTorch ≥ 2.7, instalado do índice `cu128`, configurado no `pyproject.toml`.

## Justificativa

- Suporte oficial a Blackwell (sm_120) via CUDA 12.8.
- O ecossistema da área usa PyTorch, o que facilita reproduzir baselines da literatura.

> **Revisão 2026-10-22 (ADR-0007):** com o ambiente em container Linux, a falta de GPU do TensorFlow no Windows nativo deixa de ser um argumento. A decisão se mantém pelos dois motivos acima.

## Consequências

Para a demo em tempo real, a exportação passa a ser ONNX em vez de TFLite.
