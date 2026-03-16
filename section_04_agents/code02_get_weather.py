import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()


# ─────────────────────────────────────────
# Ferramentas do agente
# ─────────────────────────────────────────

@tool
def search(query: str) -> str:
    """Busca informações gerais com base em uma consulta."""
    return f"Resultados para: {query}"


@tool
def get_weather(location: str) -> str:
    """Obtém informações de clima para uma localização."""
    return f"Clima em {location}: Ensolarado, 22°C"


# ─────────────────────────────────────────
# Inicialização do modelo e agente
# ─────────────────────────────────────────

@st.cache_resource
def build_agent():
    """Constrói e retorna o agente LangChain (cacheado pelo Streamlit)."""
    model = init_chat_model("openai:gpt-4o-mini")
    return create_agent(
        model=model,
        tools=[search, get_weather],
        system_prompt=(
            "Você é um assistente inteligente com acesso a ferramentas de "
            "busca de informações e previsão do tempo. "
            "Use a ferramenta adequada conforme a solicitação do usuário. "
            "Responda sempre em português do Brasil."
        ),
    )


# ─────────────────────────────────────────
# Interface Streamlit
# ─────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Assistente Inteligente",
        page_icon="🤖",
        layout="centered",
    )

    st.title("🤖 Assistente Inteligente")
    st.caption("Powered by LangChain + OpenAI · Ferramentas: Busca e Clima")

    # Inicializa histórico de mensagens na sessão
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibe o histórico de mensagens
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Campo de entrada do usuário
    user_input = st.chat_input("Digite sua pergunta...")

    if user_input:
        # Exibe a mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Processa a resposta do agente
        with st.chat_message("assistant"):
            with st.spinner("Processando..."):
                agent = build_agent()
                response = agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]}
                )
                answer = response["messages"][-1].content

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    # ── Sidebar com informações e controles ──
    with st.sidebar:
        st.header("⚙️ Configurações")
        st.markdown("**Ferramentas disponíveis:**")
        st.markdown("- 🔍 **Busca** — pesquisa informações gerais")
        st.markdown("- 🌤️ **Clima** — previsão do tempo por localização")
        st.divider()
        if st.button("🗑️ Limpar conversa"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
