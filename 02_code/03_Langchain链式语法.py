#导入库
from langchain_ollama import ChatOllama

#创建大模型实例
llm=ChatOllama(model="qwen3:0.6b",base_url="http://127.0.0.1:11434")