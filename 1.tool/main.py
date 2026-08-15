from pathlib import Path

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

load_dotenv()


# ============================================================
# TOOL 1: Save Research Note
# ============================================================

@tool
def save_research_note(title: str, content: str) -> str:
    """
    Save a research note as a Markdown file inside
    artifacts/save_research_note/.
    """

    project_root = Path(__file__).resolve().parent

    notes_folder = (
        project_root
        / "artifacts"
        / "save_research_note"
    )

    notes_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    safe_title = "".join(
        c if c.isalnum() or c in (" ", "-", "_") else "_"
        for c in title
    ).strip()

    file_path = notes_folder / f"{safe_title}.md"

    markdown = f"""# {title}

{content}
"""

    file_path.write_text(
        markdown,
        encoding="utf-8"
    )

    return f"Research note saved successfully at: {file_path}"


# ============================================================
# TOOL 2: Calculate Percentage Change
# ============================================================

@tool
def calculate_percentage_change(
    old_value: float,
    new_value: float
) -> float:
    """
    Calculate percentage change from old_value to new_value.
    """

    if old_value == 0:
        raise ValueError("old_value cannot be zero.")

    percentage_change = (
        (new_value - old_value) / old_value
    ) * 100

    return round(percentage_change, 2)


# ============================================================
# GEMINI 2.5 FLASH
# ============================================================

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)


# ============================================================
# CREATE DEEP AGENT
# ============================================================

agent = create_deep_agent(
    model=model,

    # Custom tools
    tools=[
        save_research_note,
        calculate_percentage_change,
    ],

    # Built-in/prebuilt tools
    # Deep Agents provides a Python execution capability.
    system_prompt="""
You are a helpful research assistant.

You have access to:
1. save_research_note - save research as a Markdown file.
2. calculate_percentage_change - calculate percentage changes.
3. Python execution - use Python when computation or code
   execution is useful.

Use tools when they are appropriate instead of trying to
perform complex calculations manually.
""",
)


# ============================================================
# TERMINAL CHATBOT
# ============================================================

print("🤖 Small Chatbot")
print("Type 'exit' to quit.\n")

messages = []

while True:
    user = input("You: ")

    if user.lower() == "exit":
        print("Bot: Goodbye! 👋")
        break

    messages.append({
        "role": "user",
        "content": user
    })

    response = agent.invoke({
        "messages": messages
    })

    messages = response["messages"]

    print("Bot:", messages[-1].content)