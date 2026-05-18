#导入库
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

#创建大模型实例
llm=ChatOllama(model="qwen3:0.6b",base_url="http://127.0.0.1:11434")

#创建提示词模板
prompt=ChatPromptTemplate.from_template('解析古诗{pome},并以JSON格式输出:{{"诗词":"...","作者":"...","核心意境":"..."}}')

#创建解析器
parser=JsonOutputParser()

#LECT 链式结构
chain= prompt | llm | parser

#大模型回答
re=chain.invoke({"pome":"结庐在人境，而无车马喧。问君何能尔？心远地自偏。采菊东篱下，悠然见南山。山气日夕佳，飞鸟相与还。此中有真意，欲辨已忘言。"})

#输出
print(re)
print("\n")
print("\n")
print(f'核心意境:{re['核心意境']}')




"""
总结：
（1）一层大括号 和 两层大括号 的区别
- {pome} = 变量 （一层大括号）
- {{内容}} = 原样输出大括号（两层大括号）

（2）遇到的问题
问题：在使用qwen3：0.6b小模型的时候可以输出做答  可是当使用qwen3.5：0.8b更大一点的模型的时候却输出不了作答

原因：小模型（0.6b）因为能力弱，只会"死记硬背"指令，所以输出更稳定
     稍大一点的模型（0.8b）因为能力强，会自作聪明多加多于内容，反而不符合解析器的严格要求。

"""