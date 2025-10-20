"""
Este módulo implementa um pipeline de processamento de texto usando LangChain para:
1. Carregar conteúdo de websites
2. Dividir em chunks
3. Criar embeddings e armazenar em uma base vetorial
4. Realizar sumarização via map-reduce
5. Responder perguntas usando RAG (Retrieval Augmented Generation)
"""

from typing import List, Dict
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_websites(urls: List[str]) -> List[Document]:
    """
    Carrega o conteúdo de múltiplos websites usando WebBaseLoader.
    
    Args:
        urls: Lista de URLs para carregar
        
    Returns:
        Lista de Documents contendo o conteúdo dos sites
    """
    loader = WebBaseLoader(urls)
    return loader.load()

def split_documents(docs: List[Document], 
                   chunk_size: int = 1200,
                   chunk_overlap: int = 150) -> List[Document]:
    """
    Divide documentos em chunks menores para processamento.
    
    Args:
        docs: Lista de documentos para dividir
        chunk_size: Tamanho máximo de cada chunk
        chunk_overlap: Quantidade de sobreposição entre chunks
        
    Returns:
        Lista de Documents divididos em chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    return splitter.split_documents(docs)

# URLs de exemplo para processamento
SAMPLE_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://langchain.readthedocs.io/en/latest/"
]

# Carrega e divide os documentos
docs = load_websites(SAMPLE_URLS)
splits = split_documents(docs)

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseRetriever

def setup_vectorstore(documents: List[Document],
                     embedding_model: str = "text-embedding-3-large",
                     k: int = 4) -> BaseRetriever:
    """
    Configura uma base vetorial FAISS com embeddings dos documentos.
    
    Args:
        documents: Lista de documentos para indexar
        embedding_model: Modelo de embeddings a usar
        k: Número de documentos similares a recuperar
        
    Returns:
        Retriever configurado para busca semântica
    """
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})

def create_prompts() -> Dict[str, ChatPromptTemplate]:
    """
    Cria os templates de prompt para sumarização e QA.
    
    Returns:
        Dicionário com os prompts configurados
    """
    return {
        "summary": ChatPromptTemplate.from_messages([
            ("system",
             "Você é um assistente que cria um resumo fiel e objetivo. "
             "Use apenas o conteúdo fornecido. Em português. Cite tópicos-chave."),
            ("human", "Conteúdo:\n\n{chunk}\n\nFaça um resumo conciso (5-8 bullets).")
        ]),
        
        "reduce": ChatPromptTemplate.from_messages([
            ("system", 
             "Você agregará múltiplos resumos parciais em um único resumo coerente, "
             "sem repetir, mantendo pontos essenciais. Em português."),
            ("human", "Resumos parciais:\n\n{bullets}\n\nProduza um resumo final de 8-12 bullets.")
        ]),
        
        "qa": ChatPromptTemplate.from_messages([
            ("system",
             "Você responde apenas com base no CONTEXTO abaixo. "
             "Se não houver evidências suficientes, diga que não sabe. "
             "Inclua 'Fontes:' listando URLs das passagens usadas.\n\n"
             "CONTEXTO:\n{context}"),
            ("human", "Pergunta: {question}")
        ])
    }

# Setup da infraestrutura
retriever = setup_vectorstore(splits)
prompts = create_prompts()

from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from typing import List, Dict, Any

def setup_llm(model: str = "gpt-4o-mini", 
             temperature: float = 0) -> ChatOpenAI:
    """
    Configura o modelo de linguagem para processamento.
    
    Args:
        model: Nome do modelo a ser usado
        temperature: Temperatura para geração (0-1)
        
    Returns:
        Instância configurada do ChatOpenAI
    """
    return ChatOpenAI(model=model, temperature=temperature)

def format_docs(docs: List[Document]) -> str:
    """
    Formata documentos para apresentação, incluindo URLs.
    
    Args:
        docs: Lista de documentos para formatar
        
    Returns:
        String formatada com conteúdo e URLs
    """
    parts = []
    for d in docs:
        url = d.metadata.get("source") or d.metadata.get("url")
        parts.append(f"[{url}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

def create_map_reduce_chain(llm: ChatOpenAI, 
                          prompts: Dict[str, ChatPromptTemplate]) -> Any:
    """
    Cria chain para sumarização map-reduce.
    
    Args:
        llm: Modelo de linguagem configurado
        prompts: Dicionário com prompts configurados
        
    Returns:
        Tuple com map_chain e reduce_chain
    """
    map_chain = (
        prompts["summary"]
        | llm
        | StrOutputParser()
    )
    
    reduce_chain = (
        prompts["reduce"]
        | llm
        | StrOutputParser()
    )
    
    return map_chain, reduce_chain

def create_rag_chain(retriever: BaseRetriever,
                    llm: ChatOpenAI, 
                    qa_prompt: ChatPromptTemplate) -> Any:
    """
    Cria chain para perguntas e respostas com RAG.
    
    Args:
        retriever: Retriever configurado
        llm: Modelo de linguagem
        qa_prompt: Template do prompt QA
        
    Returns:
        Chain configurada para QA
    """
    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

def process_documents(splits: List[Document], 
                     map_chain: Any,
                     reduce_chain: Any,
                     limit: int = 30) -> str:
    """
    Processa documentos gerando um resumo via map-reduce.
    
    Args:
        splits: Lista de documentos divididos
        map_chain: Chain para mapping
        reduce_chain: Chain para reducing
        limit: Limite de documentos a processar
        
    Returns:
        Resumo final dos documentos
    """
    partial_summaries = [
        map_chain.invoke({"chunk": d.page_content})
        for d in splits[:limit]
    ]
    
    return reduce_chain.invoke({
        "bullets": "\n".join(partial_summaries)
    })

# Setup e execução
llm = setup_llm()
map_chain, reduce_chain = create_map_reduce_chain(llm, prompts)
rag_chain = create_rag_chain(retriever, llm, prompts["qa"])

# Gera resumo
summary = process_documents(splits, map_chain, reduce_chain)
print("==== RESUMO FINAL ====\n", summary)

# Exemplo de QA
question = "Quais são os riscos ao projetar agentes autônomos?"
answer = rag_chain.invoke(question)
print("\n==== RESPOSTA ====\n", answer)
