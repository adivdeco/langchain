import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_ollama import ChatOllama

load_dotenv()

llm = ChatOllama(
    model="gpt-oss:120b-cloud",

)

print("Sending prompt to Ollama Cloud...")
response = llm.invoke("helo")

print("\nResponse:")
print(response.content)