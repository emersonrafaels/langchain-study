import rich
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

# Load environment variables from a .env file
load_dotenv()

@tool
def calculate_shipping(weight_kg: float, distance_km: float) -> str:
    """Calculates a simple shipping estimate."""
    base_rate = 12.0
    weight_cost = weight_kg * 1.8
    distance_cost = distance_km * 0.45
    total = base_rate + weight_cost + distance_cost
    return f"Estimated shipping cost: $ {total:.2f}"


model = init_chat_model("openai:gpt-4o-mini")

agent = create_agent(
    model=model,
    tools=[calculate_shipping],
    system_prompt=(
        "You are a logistics operations assistant. "
        "When the question involves shipping cost calculation, use the available tool. "
        "If there is not enough data, ask for the missing data."
    ),
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "How much is the shipping cost for 8 kg over a distance of 35 km?",
            }
        ]
    }
)

rich.print(response)

last_message = response["messages"][-1]
print("\nFINAL RESPONSE:")
rich.print(last_message.content)