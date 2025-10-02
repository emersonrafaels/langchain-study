"""
Estudo de Embeddings usando Sentence Transformers

Este exemplo demonstra como:
1. Gerar embeddings de texto usando modelos pré-treinados
2. Visualizar as dimensões e valores dos embeddings
3. Calcular similaridade entre frases
4. Comparar diferentes tipos de frases
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def print_embedding_info(texto: str, embedding: np.ndarray):
    """Exibe informações sobre o embedding de um texto"""
    print(f"\nTexto: '{texto}'")
    print(f"Dimensões do embedding: {embedding.shape}")
    print(f"Primeiros 5 valores: {embedding[:5].round(4)}")
    print(f"Magnitude (norma L2): {np.linalg.norm(embedding):.4f}")

def calcular_similaridade(texto1: str, texto2: str, model) -> float:
    """Calcula a similaridade do cosseno entre dois textos"""
    emb1, emb2 = model.encode([texto1, texto2])
    return cosine_similarity([emb1], [emb2])[0][0]

# 1. Carrega modelo pré-treinado multilíngue
print("Carregando modelo 'all-MiniLM-L6-v2'...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Exemplos com diferentes níveis de similaridade
exemplos = [
    # Par similar (mesmo significado)
    ("O gato está dormindo no sofá",
     "Um felino descansa no sofá"),
    
    # Par relacionado (mesmo objeto, ação diferente)
    ("O gato está dormindo no sofá",
     "O gato está brincando no sofá"),
    
    # Par diferente (objetos diferentes)
    ("O gato está dormindo no sofá",
     "A televisão está ligada na sala")
]

# 3. Análise de cada par
print("\n=== Análise de Similaridade ===")
for texto1, texto2 in exemplos:
    similaridade = calcular_similaridade(texto1, texto2, model)
    print(f"\nTexto 1: '{texto1}'")
    print(f"Texto 2: '{texto2}'")
    print(f"Similaridade: {similaridade:.4f}")

# 4. Análise detalhada de um embedding
texto_exemplo = "O gato está dormindo no sofá"
embedding = model.encode(texto_exemplo)
print("\n=== Análise Detalhada de Embedding ===")
print_embedding_info(texto_exemplo, embedding)

# 5. Demonstração de consistência
print("\n=== Demonstração de Consistência ===")
frases = [
    "O gato está dormindo",
    "O gato está dormindo.",  # com ponto
    "o gato está dormindo",   # sem maiúscula
    "O  gato  está  dormindo" # espaços extras
]

# Calcula similaridade entre a primeira frase e as demais
base = frases[0]
for frase in frases[1:]:
    sim = calcular_similaridade(base, frase, model)
    print(f"'{base}' vs '{frase}': {sim:.4f}")