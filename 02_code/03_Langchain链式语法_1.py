#导入库
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

#创建大模型实例
llm=ChatOllama(model="qwen3:0.6b",base_url="http://127.0.0.1:11434")

#提示词模板
#template=ChatPromptTemplate.from_messages([
#    ("system","你是一位富有文采的{role}。"),
#    ("user","请你帮我写一首关于{topic}的四句诗歌。")
#])

template=ChatPromptTemplate.from_template("以{topic}为主题，写一首四句的诗歌")

#LCET 链式语法
chain=template | llm

#结果
re=chain.invoke({"topic":"Langchain"})

print(re)
print("\n")
print("\n")
print(re.content)

"""
总结：

template=ChatPromptTemplate.from_template("以{topic}为主题，写一首四句的诗歌")
chain=template | llm
re=chain.invoke({"topic":"Langchain"})

此时template（模板）拿到的是：
    用户：以Langchain为主题，写一首四句的诗歌
"""