from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

def make_chain(temperature: float):
    """
    Create a chain with a specific temperature setting.
    """
    llm = ChatOpenAI(model=base_model, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "{question}")
    ])
    return prompt | llm | StrOutputParser()

# Load environment variables from a .env file
load_dotenv()

# Define base language model settings
base_model = "gpt-4o-mini"
system_message = "You are a helpful assistant. Please explain concepts clearly."

# Define the question and different temperatures
question = "Explain RAG in 2 sentences."
temperatures = {
    "Conservative (0.0)": 0.0,
    "Balanced (0.3)": 0.3,
    "Creative (0.7)": 0.7
}

# Generate and print responses for each temperature setting
for name, temp in temperatures.items():
    chain = make_chain(temp)
    print(f"\n=== Temperature: {name} ===")
    print(chain.invoke({"question": question}))

