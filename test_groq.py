# test_groq.py
from langchain_groq import ChatGroq
import sys
sys.path.append(".")
from config import GROQ_API_KEY, MODEL_NAME

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=MODEL_NAME
)

response = llm.invoke("Say: AgentRAG connection successful.")
print(response.content)