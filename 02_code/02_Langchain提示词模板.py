#导入库
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate  #用来创建动态提示词模板


#创建大模型实例
llm=ChatOllama(model="qwen3:0.6b",base_url="http://127.0.0.1:11434")

#创建动态提示词模板
template=ChatPromptTemplate.from_messages([
    ("system","你是一位精通多国语言的{role}"),
    ("user","请你帮我将{topic}翻译成英语。")
])

#动态填充变量(也就是将留空的变量填写想填写的文字)
prompt=template.invoke({"role":"翻译官" , "topic":"什么是LangChain？"})

#把填充好的问题直接给到大模型
re=llm.invoke(prompt)

print(re)
print("\n")
print("\n")
print(re.content)