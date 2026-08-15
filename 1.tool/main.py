import os
from pathlib import Path

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

load_dotenv()


@tool
def save_research_note(title: str, content: str) -> str:
    """
    Save a research note as a Markdown file.

    The note is saved inside:
    artifacts/save_research_note/
    """

    # Project root
    project_root = Path(__file__).resolve().parent

    # Research notes folder
    notes_folder = project_root / "artifacts" / "save_research_note"

    # Create folder if it doesn't exist
    notes_folder.mkdir(parents=True, exist_ok=True)

    # Make a safe filename
    safe_title = "".join(
        c if c.isalnum() or c in (" ", "-", "_") else "_"
        for c in title
    ).strip()

    filename = f"{safe_title}.md"

    # Final file location
    file_path = notes_folder / filename

    # Markdown content
    markdown = f"""# {title}

{content}
"""

    # Save research
    file_path.write_text(markdown, encoding="utf-8")

    return f"Research note saved successfully at: {file_path}"



model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)


agent = create_deep_agent(
    model=model,
    tools=[save_research_note],
)



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