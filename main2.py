import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel

# ==============================
# Load Environment Variables
# ==============================
load_dotenv()

# ==============================
# Create LLM
# ==============================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=3000
)

# ==============================
# Reusable Chain Builder Function
# ==============================
def create_chain(template: str, input_variables: list):
    """
    Creates a reusable LangChain chain
    """
    prompt = PromptTemplate(
        input_variables=input_variables,
        template=template
    )
    return prompt | llm | StrOutputParser()


# ==============================
# SIMPLE CHAIN
# ==============================
print("\n===== SIMPLE CHAIN =====")

simple_chain = create_chain(
    "Suggest a fancy name for a {cuisine} restaurant.",
    ["cuisine"]
)

response = simple_chain.invoke({"cuisine": "Indian"})
print("Simple Chain Output:", response)


# ==============================
# SEQUENTIAL CHAIN
# ==============================
print("\n===== SEQUENTIAL CHAIN =====")

name_chain = create_chain(
    "Suggest one fancy name for a {cuisine} restaurant.",
    ["cuisine"]
)

menu_chain = create_chain(
    "Suggest 5 menu items for {restaurant_name}. Return comma separated list.",
    ["restaurant_name"]
)

restaurant_name = name_chain.invoke({"cuisine": "Italian"})
menu = menu_chain.invoke({"restaurant_name": restaurant_name})

print("Restaurant Name:", restaurant_name)
print("Menu:", menu)


# ==============================
# CONVERSATION MEMORY
# ==============================
print("\n===== CONVERSATION MEMORY =====")

chat_history = []

chat_history.append(HumanMessage(content="Suggest a name for an Indian restaurant"))

response = llm.invoke(chat_history)
chat_history.append(AIMessage(content=response.content))

print("AI:", response.content)

chat_history.append(HumanMessage(content="Now suggest menu items for that restaurant"))

response2 = llm.invoke(chat_history)
chat_history.append(AIMessage(content=response2.content))

print("AI:", response2.content)


# ==============================
# TOOL EXAMPLE
# ==============================
print("\n===== TOOL EXAMPLE =====")

@tool
def calculate_price(price: float, tax: float) -> float:
    """Calculate final price including tax"""
    return price + (price * tax / 100)

print("Tool Example (100 + 18% tax):", calculate_price.invoke({"price": 100, "tax": 18}))


# ==============================
# STRUCTURED OUTPUT (JSON)
# ==============================
print("\n===== STRUCTURED OUTPUT (JSON) =====")

class Restaurant(BaseModel):
    name: str
    menu: list[str]

parser = JsonOutputParser(pydantic_object=Restaurant)

structured_prompt = PromptTemplate(
    input_variables=["cuisine"],
    template="""
    Suggest a restaurant name and 5 menu items for {cuisine} food.
    {format_instructions}
    """,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

structured_chain = structured_prompt | llm | parser

result = structured_chain.invoke({"cuisine": "Indian"})
print("Structured Output:", result)


# ==============================
# PARALLEL CHAIN
# ==============================
print("\n===== PARALLEL CHAIN =====")

name_chain_parallel = create_chain(
    "Suggest a fancy restaurant name for {cuisine} food.",
    ["cuisine"]
)

menu_chain_parallel = create_chain(
    "Suggest 5 popular dishes for {cuisine} cuisine. Return comma separated list.",
    ["cuisine"]
)

slogan_chain_parallel = create_chain(
    "Write a catchy slogan for a {cuisine} restaurant.",
    ["cuisine"]
)

parallel_chain = RunnableParallel(
    name=name_chain_parallel,
    menu=menu_chain_parallel,
    slogan=slogan_chain_parallel
)

result = parallel_chain.invoke({"cuisine": "Indian"})
print("Parallel Output:", result)