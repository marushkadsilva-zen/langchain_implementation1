import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.tools import tool
from pydantic import BaseModel


# Load environment variables
load_dotenv()


# DATABASE SETUP
conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()

# Long-Term Memory Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT,
    message TEXT,
    timestamp TEXT
)
""")

# Short-Term Memory Table 
cursor.execute("""
CREATE TABLE IF NOT EXISTS short_term_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT,
    message TEXT,
    timestamp TEXT
)
""")

conn.commit()


# LONG TERM MEMORY FUNCTIONS
def save_message(role, message):
    cursor.execute(
        "INSERT INTO chat_history (role, message, timestamp) VALUES (?, ?, ?)",
        (role, message, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()


def load_long_memory(limit=5):
    cursor.execute(
        "SELECT role, message FROM chat_history ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    return "\n".join([f"{r[0].upper()}: {r[1]}" for r in rows])


# SHORT TERM MEMORY FUNCTIONS
def save_short_term(role, message):
    """
    Save only last 3 AI responses in short-term memory table
    """

    cursor.execute(
        "INSERT INTO short_term_memory (role, message, timestamp) VALUES (?, ?, ?)",
        (role, message, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()

    # Keep only last 3 entries
    cursor.execute("""
        DELETE FROM short_term_memory
        WHERE id NOT IN (
            SELECT id FROM short_term_memory
            ORDER BY id DESC
            LIMIT 3
        )
    """)
    conn.commit()


def load_short_memory():
    cursor.execute("""
        SELECT role, message FROM short_term_memory
        ORDER BY id ASC
    """)
    rows = cursor.fetchall()
    return "\n".join([f"{r[0].upper()}: {r[1]}" for r in rows])


def view_history():
    print("\n===== DATABASE CHAT HISTORY =====")
    cursor.execute("SELECT role, message, timestamp FROM chat_history")
    rows = cursor.fetchall()
    for row in rows:
        print(f"[{row[2]}] {row[0].upper()}: {row[1]}")


# Create LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=3000
)

parser = StrOutputParser()


# MEMORY STACK PROMPT TEMPLATE
memory_prompt = PromptTemplate(
    input_variables=[
        "system_prompt",
        "core_memory",
        "short_memory",
        "long_memory",
        "user_input"
    ],
    template="""
{system_prompt}

CORE MEMORY:
{core_memory}

SHORT TERM MEMORY:
{short_memory}

LONG TERM MEMORY:
{long_memory}

USER:
{user_input}

ASSISTANT:
"""
)

memory_chain = memory_prompt | llm | parser

# SIMPLE CHAIN
print("\n===== SIMPLE CHAIN =====")

prompt = PromptTemplate(
    input_variables=["cuisine"],
    template="Suggest a fancy name for a {cuisine} restaurant."
)

chain = prompt | llm | parser
response = chain.invoke({"cuisine": "Indian"})
print("Simple Chain Output:", response)


# SEQUENTIAL CHAIN
print("\n===== SEQUENTIAL CHAIN =====")

name_prompt = PromptTemplate(
    input_variables=["cuisine"],
    template="Suggest one fancy name for a {cuisine} restaurant."
)

menu_prompt = PromptTemplate(
    input_variables=["restaurant_name"],
    template="Suggest 5 menu items for {restaurant_name}. Return comma separated list."
)

name_chain = name_prompt | llm | parser
menu_chain = menu_prompt | llm | parser

restaurant_name = name_chain.invoke({"cuisine": "Italian"})
menu = menu_chain.invoke({"restaurant_name": restaurant_name})

print("Restaurant Name:", restaurant_name)
print("Menu:", menu)



# MEMORY STACK CONVERSATION
print("\n===== MEMORY STACK CONVERSATION =====")

system_prompt = "You are an expert Indian restaurant consultant."
core_memory = "Always give premium, creative and business-friendly suggestions."

user_input = "Suggest a name for an Indian restaurant"

# Save user to long-term memory
save_message("user", user_input)

# Load memory layers
long_memory = load_long_memory()
short_memory = load_short_memory()

# Generate AI response
response = memory_chain.invoke({
    "system_prompt": system_prompt,
    "core_memory": core_memory,
    "short_memory": short_memory,
    "long_memory": long_memory,
    "user_input": user_input
})

print("AI:", response)

# Save AI response
save_message("ai", response)        
save_short_term("ai", response)   


# TOOL EXAMPLE
print("\n===== TOOL EXAMPLE =====")

@tool
def calculate_price(price: float, tax: float) -> float:
    """Calculate final price including tax percentage."""
    return price + (price * tax / 100)

print("Tool Example (100 + 18% tax):",
      calculate_price.invoke({"price": 100, "tax": 18}))


# STRUCTURED OUTPUT
print("\n===== STRUCTURED OUTPUT (JSON) =====")

class Restaurant(BaseModel):
    name: str
    menu: list[str]

json_parser = JsonOutputParser(pydantic_object=Restaurant)

structured_prompt = PromptTemplate(
    input_variables=["cuisine"],
    template="""
Suggest a restaurant name and 5 menu items for {cuisine} food.
{format_instructions}
""",
    partial_variables={"format_instructions": json_parser.get_format_instructions()}
)

structured_chain = structured_prompt | llm | json_parser
result = structured_chain.invoke({"cuisine": "Indian"})

print("Structured Output:", result)


# PARALLEL CHAIN
print("\n===== PARALLEL CHAIN =====")

name_prompt = PromptTemplate(
    input_variables=["cuisine"],
    template="Suggest a fancy restaurant name for {cuisine} food."
)

menu_prompt = PromptTemplate(
    input_variables=["cuisine"],
    template="Suggest 5 popular dishes for {cuisine} cuisine."
)

slogan_prompt = PromptTemplate(
    input_variables=["cuisine"],
    template="Write a catchy slogan for a {cuisine} restaurant."
)

name_chain = name_prompt | llm | parser
menu_chain = menu_prompt | llm | parser
slogan_chain = slogan_prompt | llm | parser

parallel_chain = RunnableParallel(
    name=name_chain,
    menu=menu_chain,
    slogan=slogan_chain
)

result = parallel_chain.invoke({"cuisine": "Indian"})
print("Parallel Output:", result)

# VIEW DATABASE HISTORY
view_history()

conn.close()