#导入库
from langchain_ollama import ChatOllama
from langchain_core.tools import tool


#定义工具
@tool
def multiply(a:int , b:int) ->int:
    """用于计算两个数字的乘法"""
    return a*b

#工具列表
tools=[multiply]

#创建大模型实例,并且带上工具功能
llm=ChatOllama(model='qwen3.5:0.8b',base_url='http://127.0.0.1:11434').bind_tools(tools)

#对大模型提问（大模型会分析问题，判断需要调用的工具，并生成调用请求）
re=llm.invoke("11890*2224等于几？")

#从模型返回的结果，提取工具调用信息
tool_call=re.tool_calls[0]

#真正执行工具：调用multiply函数，传入模型提取的参数
result=multiply.invoke(tool_call['args'])

print(result)
print('\n')
print('\n')
print(f'11890*2224 等于 {result}')




"""
总结
1、大模型调用工具
大白话：实际上大模型调用工具就是在程序员写了一个工具函数，然后大模型通过去读懂prompt，然后把prompt里面的跟函数的输入相对应的取值填写进去，程序员再使用工具才可以实现。

步骤：
1、程序员先写好工具函数
2、大模型看懂问题，从问题中抠出数字，填写函数的参数里
3、模型只做一件事情：告诉你现在正在调用哪个函数，赋值进去
4、最后必须由程序员写代码去真正调用这个函数，才能算出结果

"""