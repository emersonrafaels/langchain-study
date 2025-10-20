# Exemplos de Uso de Embeddings

Este projeto demonstra aplicações práticas de embeddings de texto usando Python, incluindo busca semântica, agrupamento de textos e detecção de duplicatas.

## 🚀 Funcionalidades

O código implementa três funcionalidades principais:

1. **Busca Semântica**: Encontra documentos semanticamente similares a uma consulta
2. **Agrupamento de Textos**: Agrupa textos similares usando K-means
3. **Detecção de Duplicatas**: Identifica textos muito similares

## 💻 Como Usar

### Pré-requisitos

```bash
pip install sentence-transformers scikit-learn numpy
```

### Exemplos de Uso

1. **Busca Semântica**:
```python
processor = TextEmbeddingProcessor()
documentos = ["Python é uma linguagem", "JavaScript é muito usado"]
resultados = processor.busca_semantica("Python programação", documentos)
```

2. **Agrupamento**:
```python
textos = ["O café está quente", "O chá está frio"]
grupos, score = processor.agrupar_textos(textos, n_grupos=2)
```

3. **Detecção de Duplicatas**:
```python
frases = ["O produto chegou", "A entrega foi feita"]
duplicatas = processor.detectar_duplicatas(frases, threshold=0.8)
```

## 🔧 Estrutura do Código

- `TextEmbeddingProcessor`: Classe principal que gerencia as operações
- `SearchResult`: Dataclass para resultados de busca
- Usa o modelo `all-MiniLM-L6-v2` por padrão
- Implementa cache de embeddings para melhor performance

## 📝 Notas Técnicas

- Utiliza `sentence-transformers` para geração de embeddings
- Implementa similaridade por cosseno para comparações
- Usa K-means para agrupamento com score silhouette
- Cache automático de embeddings para otimização

## 🤝 Contribuindo

Sinta-se à vontade para contribuir com melhorias através de PRs ou sugerir novas funcionalidades.
