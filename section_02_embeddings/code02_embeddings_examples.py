"""
Exemplos Práticos de Uso de Embeddings

Este código demonstra aplicações práticas de embeddings:
1. Busca semântica
2. Agrupamento de textos similares
3. Detecção de duplicatas aproximadas
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Classe para armazenar resultados da busca"""

    texto: str
    score: float


class TextEmbeddingProcessor:
    """Classe principal para processamento de embeddings"""

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None
    ):
        """
        Inicializa o processador de embeddings

        Args:
            model_name: Nome do modelo de embedding
            cache_dir: Diretório para cache de embeddings
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        logger.info(f"Carregando modelo {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_cache = {}

    def _get_embedding(self, texto: str) -> np.ndarray:
        """Obtém embedding com cache"""
        if texto in self.embedding_cache:
            return self.embedding_cache[texto]

        embedding = self.model.encode([texto])[0]
        self.embedding_cache[texto] = embedding
        return embedding

    def busca_semantica(
        self, query: str, documentos: List[str], top_k: int = 3
    ) -> List[SearchResult]:
        """
        Realiza busca semântica

        Args:
            query: Texto da consulta
            documentos: Lista de documentos para busca
            top_k: Número de resultados

        Returns:
            Lista de SearchResult ordenada por relevância
        """
        if not documentos:
            raise ValueError("Lista de documentos vazia")
        if top_k < 1:
            raise ValueError("top_k deve ser >= 1")

        query_emb = self._get_embedding(query)
        docs_emb = np.vstack([self._get_embedding(doc) for doc in documentos])

        similaridades = cosine_similarity([query_emb], docs_emb)[0]
        indices = np.argsort(similaridades)[::-1][:top_k]

        return [SearchResult(documentos[i], float(similaridades[i])) for i in indices]

    def agrupar_textos(
        self, textos: List[str], n_grupos: int = 2
    ) -> Tuple[List[int], float]:
        """
        Agrupa textos similares usando K-means

        Args:
            textos: Lista de textos para agrupar
            n_grupos: Número de grupos

        Returns:
            Tuple com grupos e score silhouette
        """
        embeddings = np.vstack([self._get_embedding(texto) for texto in textos])
        kmeans = KMeans(n_clusters=n_grupos, random_state=42)
        grupos = kmeans.fit_predict(embeddings)

        # Avalia qualidade do agrupamento
        silhouette_avg = silhouette_score(embeddings, grupos)
        logger.info(f"Silhouette Score: {silhouette_avg:.4f}")

        return grupos, silhouette_avg

    def detectar_duplicatas(
        self, frases: List[str], threshold: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """
        Detecta frases similares

        Args:
            frases: Lista de frases
            threshold: Limiar de similaridade

        Returns:
            Lista de tuplas (frase1, frase2, similaridade)
        """
        resultados = []
        for i in range(len(frases)):
            emb_i = self._get_embedding(frases[i])
            for j in range(i + 1, len(frases)):
                emb_j = self._get_embedding(frases[j])
                sim = float(cosine_similarity([emb_i], [emb_j])[0][0])
                if sim > threshold:
                    resultados.append((frases[i], frases[j], sim))
        return resultados


# Exemplo de uso
if __name__ == "__main__":
    processor = TextEmbeddingProcessor()

    # 1. Exemplo de Busca Semântica
    logger.info("\n=== Busca Semântica ===")
    documentos = [
        "Python é uma linguagem de programação de alto nível",
        "JavaScript é muito usado no desenvolvimento web",
        "Machine Learning usa algoritmos de aprendizado",
        "Deep Learning é um subcampo do Machine Learning",
        "HTML e CSS são fundamentais para websites",
    ]

    query = "Inteligência Artificial e aprendizado de máquina"
    logger.info(f"\nBusca por: '{query}'")
    resultados = processor.busca_semantica(query, documentos)
    for res in resultados:
        logger.info(f"Score: {res.score:.4f} | Doc: '{res.texto}'")

    # 2. Exemplo de Agrupamento
    logger.info("\n=== Agrupamento de Textos ===")
    textos = [
        "O café está quente",
        "O chá está frio",
        "Programação em Python",
        "Desenvolvimento com Python",
        "A bebida está gelada",
        "Código em Python",
    ]

    logger.info("Executando agrupamento...")
    grupos, silhouette = processor.agrupar_textos(textos, n_grupos=2)
    for i, (texto, grupo) in enumerate(zip(textos, grupos)):
        logger.info(f"Texto {i+1}: '{texto}' -> Grupo {grupo}")

    # 3. Exemplo de Detecção de Duplicatas
    logger.info("\n=== Detecção de Duplicatas ===")
    frases = [
        "O produto chegou no prazo",
        "A entrega foi feita dentro do prazo",
        "Estou muito satisfeito com o produto",
        "O item chegou no tempo previsto",
        "Não gostei do atendimento",
    ]

    logger.info("Detectando duplicatas...")
    duplicatas = processor.detectar_duplicatas(frases)
    for f1, f2, sim in duplicatas:
        logger.info(f"\nSimilaridade: {sim:.4f}")
        logger.info(f"Frase 1: '{f1}'")
        logger.info(f"Frase 2: '{f2}'")
