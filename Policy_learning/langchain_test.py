from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """ Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="deepseek-r1")

chain = prompt | model
print(f"Invoking chat: {chain}")
response = chain.invoke({"question": "What is LangChain?"})
print(response)
