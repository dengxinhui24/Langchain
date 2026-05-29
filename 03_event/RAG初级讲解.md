- 导入包
```python
from langchain_ollama import ChatOllama,OllamaEmbeddings                 #ChatOllama用来创建大模型实例  OllamaEmbeddings用来 文本->向量 
from langchain_community.document_loaders import PyPDFLoader             #PyPDFLoader用来读取和解析PDF文件内容的工具
from langchain_text_splitters import RecursiveCharacterTextSplitter      #RecursiveCharacterTextSplitter用来把PDF切成一段段小文本块
from langchain_community.vectorstores import Chroma                      #Chroma用来做向量数据库          
from langchain_community.retrievers import BM25Retriever                 #BM25Retriever是一个基于关键词的传统检索工具
from langchain_classic.retrievers import EnsembleRetriever               #EnsembleRetriever是一个多检索器的容器+结果融合排序器
from sentence_transformers import CrossEncoder                           #CrossEncoder对于检索结果“二次打分“的模型 （需要结合重排序使用）
from langchain_core.prompts import ChatPromptTemplate                    #ChatPromptTemplate是一个定义提问格式的提示词模板工具
from langchain_core.output_parsers import StrOutputParser                #StrOutputParser字符串输出解析器（把大模型返回的消息对象，直接解析成纯文本字符串）
from langchain_core.runnables import RunnablePassthrough,RunnableLambda  #RunnableLambda把用户写的普通函数，包装成Langchain链能识别的组件，RunnablePassthrough把输入原封不动地往下传，相当于链里的“管道直通器”
```

- 创建大模型实例
```python
llm=ChatOllama(model="qwen3:0.6b",base_url="http://127.0.0.1:11434")
```
<img width="623" height="27" alt="image" src="https://github.com/user-attachments/assets/0ca3c1ab-d439-49ba-91a4-88fbd3975558" />

- 导入PDF文档
```python
loader=PyPDFLoader(r"软件与人工智能学院本科生学业预警实施办法.pdf")  #这是一个对象实例
docs=loader.load() #现在doc就是这个PDF的内容
```
<img width="1102" height="155" alt="image" src="https://github.com/user-attachments/assets/8cb9af72-17e4-4708-97ca-a96c2eb50b14" />

- 切分文档
```python
text_split=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50) #切分，每一块最大300字，块与块之间重复覆盖50字 （是一个文本分割器对象）
doc=text_split.plit_documents(docs)
```
<img width="1096" height="123" alt="image" src="https://github.com/user-attachments/assets/5d403045-658c-4067-89b4-3735d7465616" />

- 词嵌入（文本->向量）
```python
embeddings=OllamaEmbeddings(model='bge-m3:latest',base_url='http://127.0.0.1:11434') #创建词嵌入模型实例
```
<img width="1099" height="54" alt="image" src="https://github.com/user-attachments/assets/f81c2662-6349-4590-9f44-288d466ffcef" />

- 建立向量库
```python
vectorstore=Chroma.from_documents(documents=doc,embedding=embeddings,persist_directory="chroma_db") #persist_directory是指存放的位置
```

- 从向量库中构建检索器
```python
vectore_retriever=vectorstore.as_retriever(search_kwargs={'k':6}) #search_kwargs={'k':6} 是指每次检索时，返回与问题向量最接近的前6个文本块
```
<img width="1093" height="39" alt="image" src="https://github.com/user-attachments/assets/dc192303-de44-445f-8770-2d87f539fff6" />

- 构建BM25检索器
```python
bm25_retriever=BM25Retriever.from_documents(doc)
bm25_retriever.k=6  #设置返回与问题最接近的6个文本块
```
<img width="586" height="33" alt="image" src="https://github.com/user-attachments/assets/f399df53-3051-4fff-b395-62b74a96708a" />


- 构建混合检索器
```python
ensemble_retriever=EnsembleRetriever(
    retrievers=[vectore_retriever,bm25_retriever],
    weights=[0.5,0.5]
) #weights是指权重，也就是说这两个检索器的结果各占一半，从而形成一个总的检索结果
```
<img width="944" height="53" alt="image" src="https://github.com/user-attachments/assets/f030cfd1-3431-4dbc-8e51-8ccafad990cb" />

- 构建重排序打分器
```python
reranker=CrossEncoder(r"D:\我的备份\课件（大三下）\智能应用系统设计\hugging faces_model\BAAI-bge-reranker-base")  
#现在这个reranker是一个构造器，只是用来做一个打分工作的，其输入为（问题+文本）   输出为（一个分数  0-1之间）   !!并且这个模型路径可以用Huggingface官方的直通地址，我这里是转化为了下载好的模型地址
```
- 定义重排序函数
```python

```
