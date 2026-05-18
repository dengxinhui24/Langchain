#导入库
from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
import os

#创建大模型实例
#os.environ['OLLAMA_HOST']='http://127.0.0.1:11434'
#llm=OllamaLLM(model="qwen3:0.6b")

#创建大模型实例
llm=ChatOllama(model="qwen3:0.6b",base_url="http://127.0.0.1:11434")

#使用llm大模型进行问答
re=llm.invoke("请用一句话介绍一下LangChain")

#输出大模型回答
print(re)
print()
print()
print(re.content)

"""
总结

本地大模型调用有OllamaLLM 和 ChatOllama两种类型：

   OllamaLLM : 适用于简单的文本补全任务，输入输出都是字符串。

   ChatOllama ：适用于聊天对话场景，支持多轮对话、系统提示等，返回结构化的消息对象。

   大白话讲：OllamaLLM 输出是"字符串"       ChatOllama输出是"元数据"（包含了对话角色，和内容等信息）。
"""
