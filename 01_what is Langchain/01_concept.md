# Langchain的基本概念
## 1.什么是Langchain
Langchain 是一套用于构建AI智能体（AI Agent）和 大语言模型（LLM）应用的开发框架。

## 2.Langchain的核心流程
简化大语言模型应用开发流程。

## 3.Langchain的核心组件
- 大语言模型（LLM）  ->  连接OpenAI、Claude、Gemini、qwen3：0.6b等大模型
- 文档处理           ->  加载、分割、嵌入、检索（RAG基础）
- 向量存储           ->  连接向量数据库实现RAG
- 提示词管理         ->  管理Prompt模板
- 链（Chains）       ->  构建多步骤AI工作流（例如： ）
- 记忆（Memory）     ->  在多轮对话中保持上下文状态
- 智能体(Agents)     ->  让LLM自主决策、动态调用工具

## 4.Langchain的应用场景
- 智能聊天机器人
具备多轮对话记忆，能调用外部工具（如查天气、查订单、发邮件）的聊天助手。

- RAG知识库问答
将私有文档（PDF、网页、数据库）向量化存储，让模型能够基于这些文档回答问题，附带引用来源。

- Agent自动化助手
模型自主规划任务步骤，按需调用不同的工具，完成复杂的多步操作，如“帮我整理上周的销售数据并生成报告”。

- 数据提取与分析
从非结构化文本中提取结构化信息（如从合同扫描件中提取关键字段），或让模型生成数据分析结论。

## 5.Langchain的生态家族
- LangChain ： 核心框架（Python / Javascript）
- LangGraph ： 构建复杂多步骤工作流（状态图机制）
- LangServe ： 将chain/Agent部署为Rest API
- LangSmith ： 调试、监控、评估LLM应用的全生命周期平台

## 6.使用Langchain的准备工作
- Python基础语法（函数、类、类型注解）
- 能够使用pip安装python包
- 了解基本的命令行操作
- 可以使用ollama本地部署的模型