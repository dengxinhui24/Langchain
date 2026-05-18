#导入库
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS  #向量库

#创建Embedding模型实例
embedding=OllamaEmbeddings(model="qwen3-embedding:0.6b",base_url='http://127.0.0.1:11434')

#准备文本数据
texts=[
   "张三是一位Python开发工程师，拥有5年开发经验。",
    "李四在2025年获得了全国大学生人工智能大赛一等奖。",
    "李四是前端开发，擅长React和Vue 框架。",
    "张三毕业于广州软件学院计算机科学与技术专业。",
    "张三擅长智能体和大模型应用开发。"
]

#创建向量库（将文本数据转化为数字向量并且存储）
vectorstore=FAISS.from_texts(texts,embedding)

#定义检索器（Retriever）
retriever = vectorstore.as_retriever(search_kwargs={'k':1})  #k=1表示只取最相关的一个结果

#执行检索
prompt="张三是哪个学校的"

re=retriever.invoke(prompt)

print(re[0].page_content)



"""
总结
（1）FAISS是什么？

大白话：专门用来在一大堆向量里，超快找出"最像"的那些向量

特点(不是“数据库”，而是向量搜索引擎)：
1、处理高维向量（比如768维，1536维）
2、支持百万 / 千万 / 亿级向量
3、毫秒级查询（比传统数据库快几百倍）
4、可 CPU / GPU 加速

（2）什么是向量？

大白话：一句话，一段文本  经过->embedding模型 ->变成了 一串数字（比如768个浮点数） ：这串数字 = 向量
 
特点： 意思越接近的文本，向量距离越近。
    
（3）FAISS检索函数返回的结果永远是一个列表（哪怕只有一个元素）。


（4）pip安装FAISS的时候有两种：
        CPU： pip install faiss-cpu
        GPU:  pip install faiss-gpu

"""