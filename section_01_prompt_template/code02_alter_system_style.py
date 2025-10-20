from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

def make_chain(system_msg: str):
    
    """
    Create a chain with a specific system message style.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", "{question}")
    ])
    return prompt | llm | StrOutputParser()

# Load environment variables from a .env file
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define the question and different system styles
question = "Explain RAG in 2 sentences."

styles = {
    "professor": "You are a university professor, objective and technical.",
    "consultant": "You are a business consultant who speaks with examples.",
    "reviewer":   "You are a reviewer who returns answers in bullet points."
}

# Generate and print responses for each style
for name, system_msg in styles.items():
    # Create the chain with the specified system message
    chain = make_chain(system_msg)
    print(f"\n=== Style: {name} ===")
    print(chain.invoke({"question": question}))

