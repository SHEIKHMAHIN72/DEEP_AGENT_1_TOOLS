import os 
from dotenv import load_dotenv

from deepagents import create_deep_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

#create an deep agent 
agent = create_deep_agent(
    model=model 
)

print("🤖 Small Chatbot")
print("Type 'exit' to quit.\n")

messages = []
while True:
    user = input("YOU:")

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