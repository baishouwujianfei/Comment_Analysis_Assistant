# 功能说明：实现使用langchain框架，使用LCEL构建一个完整的LLM应用程序用于RAG知识库的查询，并使用fastapi进行发布
# 包含：langchain框架的使用，langsmith跟踪检测

# 相关依赖库
# pip install langchain langchain-openai langchain-chroma

import os
import re
import json
import asyncio
import uuid
import time
import logging
import sys
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from langchain_openai import ChatOpenAI
from tools.getConfig import GetConfig
from tools.argument_parser import ArgumentParser
from tools import excelSplit
import chardet
import pandas as pd

# prompt模版
from langchain_core.prompts import PromptTemplate
# 配置可配置字段
from langchain_core.runnables import ConfigurableFieldSpec
# 定义聊天提示模板，以及占位符替换
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 用于运行带有消息历史的可运行对象
from langchain_core.runnables.history import RunnableWithMessageHistory
# 用于处理和存储对话历史
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.schema import ChatMessage
# 部署REST API相关
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import FastAPI, File, UploadFile  # 确保从FastAPI导入UploadFile
import uvicorn
# embedding函数
from tools import connetDB
# 向量数据库chroma相关
from langchain_chroma import Chroma
# openai的向量模型
from langchain_openai import OpenAIEmbeddings
# RAG相关
from langchain_core.runnables import RunnablePassthrough


# 设置langsmith环境变量
# os.environ["LANGCHAIN_TRACING_V2"] = "false"
# os.environ["LANGCHAIN_API_KEY"] = "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"


# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [Line: %(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

argument_parser = ArgumentParser()
args = argument_parser.parse_arguments()
config = GetConfig()  
config.initialize(args)
# 向量数据库chromaDB设置相关 根据自己的实际情况进行调整
CHROMADB_DIRECTORY = "./chromaDB"  # chromaDB向量数据库的持久化路径
CHROMADB_COLLECTION_NAME = "demo001"  # 待查询的chromaDB向量数据库的集合名称

# prompt模版设置相关 根据自己的实际情况进行调整
PROMPT_TEMPLATE_TXT = "prompt_template.txt"

# 模型设置相关  根据自己的实际情况进行调整
API_TYPE = "openai"  # openai:调用gpt模型

# openai模型相关配置 根据自己的实际情况进行调整
OPENAI_API_BASE = config.openai['openai_api_base']
OPENAI_EMBEDDING_API_KEY = config.openai['openai_api_key']
OPENAI_CHAT_API_KEY = config.openai['openai_api_key']
OPENAI_CHAT_MODEL = config.openai['chat_model']
OPENAI_EMBEDDING_MODEL = config.openai['embedding_model']

# API服务设置相关  根据自己的实际情况进行调整
HOST = config.api_service['host']
PORT = config.api_service['port']  # 服务访问的端口

# 实例化全局变量 全局调用
# query_content = ''   # 将chain中传递的用户输入的信息赋值到query_content
model = None  # 使用的LLM模型
embeddings = None  # 使用的Embedding模型
vectorstore = None  # 向量数据库实例
prompt = None  # prompt内容
chain = None  # 定义的chain
vector_db = None # 向量数据库实例



# 定义Message类
class Message(BaseModel):
    role: str
    content: str

# 定义ChatCompletionRequest类
class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    userId: Optional[str] = None
    conversationId: Optional[str] = None
    history_limit: Optional[int] = 0

# 定义ChatEmbeddingRequest类
# class ChatEmbeddingRequest(BaseModel):
#     file_path: UploadFile = File(...)
#     userId: Optional[str] = None
    



# 定义ChatCompletionResponseChoice类
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

# 定义ChatCompletionResponse类
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    system_fingerprint: Optional[str] = None


# 获取prompt在chain中传递的prompt最终的内容
def getPrompt(prompt):
    logger.info(f"最后给到LLM的prompt的内容: {prompt}")
    return prompt


# 获取历史对话
# 根据用户ID和会话ID获取SQL数据库中的聊天历史
# 该函数返回一个SQLChatMessageHistory对象，用于存储特定用户和会话的历史记录
def get_session_history(user_id: str, conversation_id: str, history_limit: int):
    try:
        # 初始化SQLChatMessageHistory对象
        history = SQLChatMessageHistory(f"{user_id}--{conversation_id}", "sqlite:///memory.db")
        
        # 如果历史对话数量为0，则不加载历史对话
        if history_limit == 0:  
            logger.info("历史对话数量为0，不加载历史对话")
            history.clear()
            return history
        else:
            # 获取所有历史消息
            all_messages = history.messages
            # 只保留最近10条历史消息的content部分
            recent_messages_content = [message.content for message in all_messages[-history_limit:]]
            # logger.info(f"recent_messages_content: \n{recent_messages_content}")
            # 清空当前历史，并添加最近10条content
            history.clear()
            for i, content in enumerate(recent_messages_content):
                if i > 0 and history.messages[i-1].role == 'user':
                    role = 'assistant'
                else:
                    role = 'user'
                message = ChatMessage(content=content, role=role)
                history.add_message(message)
            print(history.messages)
            return history
    except Exception as e:
        logger.error(f"获取历史对话失败: {e}")
        return None
    

# 格式化响应，对输入的文本进行段落分隔、添加适当的换行符，以及在代码块中增加标记，以便生成更具可读性的输出
def format_response(response):
    # 使用正则表达式 \n{2, }将输入的response按照两个或更多的连续换行符进行分割。这样可以将文本分割成多个段落，每个段落由连续的非空行组成
    paragraphs = re.split(r'\n{2,}', response)
    # 空列表，用于存储格式化后的段落
    formatted_paragraphs = []
    # 遍历每个段落进行处理
    for para in paragraphs:
        # 检查段落中是否包含代码块标记
        if '```' in para:
            # 将段落按照```分割成多个部分，代码块和普通文本交替出现
            parts = para.split('```')
            for i, part in enumerate(parts):
                # 检查当前部分的索引是否为奇数，奇数部分代表代码块
                if i % 2 == 1:  # 这是代码块
                    # 将代码块部分用换行符和```包围，并去除多余的空白字符
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            # 将分割后的部分重新组合成一个字符串
            para = ''.join(parts)
        else:
            # 否则，将句子中的句点后面的空格替换为换行符，以便句子之间有明确的分隔
            para = para.replace('. ', '.\n')
        # 将格式化后的段落添加到formatted_paragraphs列表
        # strip()方法用于移除字符串开头和结尾的空白字符（包括空格、制表符 \t、换行符 \n等）
        formatted_paragraphs.append(para.strip())
    # 将所有格式化后的段落用两个换行符连接起来，以形成一个具有清晰段落分隔的文本
    return '\n\n'.join(formatted_paragraphs)


# 定义了一个异步函数 lifespan，它接收一个FastAPI应用实例app作为参数。这个函数将管理应用的生命周期，包括启动和关闭时的操作
# 函数在应用启动时执行一些初始化操作，如设置搜索引擎、加载上下文数据、以及初始化问题生成器
# 函数在应用关闭时执行一些清理操作
# @asynccontextmanager 装饰器用于创建一个异步上下文管理器，它允许你在 yield 之前和之后执行特定的代码块，分别表示启动和关闭时的操作
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    # 申明引用全局变量，在函数中被初始化，并在整个应用中使用
    global model, embeddings, vectorstore, prompt, chain, vector_db, API_TYPE, CHROMADB_DIRECTORY, CHROMADB_COLLECTION_NAME, PROMPT_TEMPLATE_TXT, PROMPT_TEMPLATE_TXT_SYSTEM, PROMPT_TEMPLATE_TXT_USER, with_message_history
    global OPENAI_API_BASE, OPENAI_CHAT_API_KEY, OPENAI_CHAT_MODEL, OPENAI_EMBEDDING_API_KEY, OPENAI_EMBEDDING_MODEL
    # 根据自己实际情况选择调用model和embedding模型类型
    try:
        logger.info("正在初始化模型、实例化Chroma对象、提取prompt模版、定义chain...")
        # （1）根据API_TYPE选择初始化对应的模型
        # 实例化一个ChatOpenAI客户端对象
        model = ChatOpenAI(
            base_url=OPENAI_API_BASE,# 请求的API服务地址
            api_key=OPENAI_CHAT_API_KEY,# API Key
            model=OPENAI_CHAT_MODEL,# 本次使用的模型
            # temperature=0,# 发散的程度，一般为0
            # timeout=None,# 服务请求超时
            # max_retries=2,# 失败重试最大次数
        )
        # 实例化embeddings处理模型
        embeddings = OpenAIEmbeddings(
            base_url=OPENAI_API_BASE,# 请求的API服务地址
            api_key=OPENAI_EMBEDDING_API_KEY,# API Key
            model=OPENAI_EMBEDDING_MODEL,
            )

        # （2）实例化Chroma对象
        # 根据自己的实际情况调整persist_directory和collection_name
        vectorstore = Chroma(persist_directory=CHROMADB_DIRECTORY,
                             collection_name=CHROMADB_COLLECTION_NAME,
                             embedding_function=embeddings,
                             )
        vector_db = connetDB.MyVectorDBConnector(CHROMADB_COLLECTION_NAME, excelSplit.batch_embed_texts)
        # （3）提取prompt模版
        
        # 指定UTF-8编码打开并读取文件
        with open(PROMPT_TEMPLATE_TXT, "r", encoding="utf-8") as f:
            template_content_user = f.read()
        prompt_template = PromptTemplate.from_template(template_content_user)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system","你是一个商品评论分析专家。你的任务是根据下述给定的已知信息回答用户问题。"),
                MessagesPlaceholder(variable_name="history"),
                ("human", prompt_template.template)
            ]
        )


        # （4）定义chain
        # 将RAG检索放到LangChain的LCEL的chain中执行
        # 这段代码是使用Langchain框架中的`as_retriever`方法创建一个检索器对象
        # LangChain VectorStore对象不是 Runnable 的子类，因此无法集成到LangChain的LCEL的chain中
        # LangChain Retrievers是Runnable，实现了一组标准方法可集成到LCEL的chain中
        # `vectorstore`是一个向量存储对象，用于存储和检索文本数据
        # `as_retriever`方法将向量存储对象转换为一个检索器对象，该对象可以用于搜索与给定查询最相似的文本
        # `search_type`参数设置为"similarity"，表示使用相似度搜索算法
        # `search_kwargs`参数是一个字典，包含搜索算法的参数，这里的`k`参数设置为5，表示只返回与查询最相似的5个结果
        # retriever = vectorstore.as_retriever(
        #     search_type="similarity",
        #     search_kwargs={"k": 5},
        # )
        # 定义chain
        # 先构建prompt模版，将用户输入消息直接赋值给prompt模版中的{query}
        # retriever返回值赋值给prompt模版中的{context}
        # 将完整的prompt给到model执行
        chain = prompt | getPrompt | model
        
        # 处理带有消息历史Chain  将可运行的链与消息历史记录功能结合
        # RunnableWithMessageHistory允许在运行链时携带消息历史
        # 实例化的with_message_history是一个配置了消息历史的可运行对象，使用get_session_history来获取历史记录
        # ConfigurableFieldSpec定义了用户ID和会话ID的配置字段，使得这些字段在运行时可以被动态传递
        with_message_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="query",
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="User ID",
                    description="Unique identifier for the user.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="conversation_id",
                    annotation=str,
                    name="Conversation ID",
                    description="Unique identifier for the conversation.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="history_limit",
                    annotation=str,
                    name="History Limit",
                    description="History Message Limit.",
                    default=0,
                    is_shared=True,
                ),
            ],
        )

        logger.info("初始化完成")

    except Exception as e:
        logger.error(f"初始化过程中出错: {str(e)}")
        # raise 关键字重新抛出异常，以确保程序不会在错误状态下继续运行
        raise

    # yield 关键字将控制权交还给FastAPI框架，使应用开始运行
    # 分隔了启动和关闭的逻辑。在yield 之前的代码在应用启动时运行，yield 之后的代码在应用关闭时运行
    yield
    # 关闭时执行
    logger.info("正在关闭...")


# lifespan 参数用于在应用程序生命周期的开始和结束时执行一些初始化或清理工作
app = FastAPI(lifespan=lifespan)


# POST请求接口，与大模型进行知识问答
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # 申明引用全局变量，在函数中被初始化，并在整个应用中使用
    if not model or not embeddings or not vectorstore or not prompt or not chain:
        logger.error("服务未初始化")
        raise HTTPException(status_code=500, detail="服务未初始化")

    try:
        logger.info(f"收到聊天完成请求: {request}")
        query_prompt = request.messages[-1].content
        logger.info(f"用户问题是: {query_prompt}")

        # 进行本地知识库检索
        # retriever = vectorstore.similarity_search(
        #     query=query_prompt,
        #     k=3,
        # )
        
        search_result = vector_db.search(query=query_prompt, top_n=3, user_id=request.userId)
        retriever = search_result["documents"]
        # 调用chain进行查询
        result = with_message_history.invoke(
            {"query": query_prompt,"context": retriever},
            config={"configurable": {"user_id": request.userId, "conversation_id": request.conversationId, "history_limit": request.history_limit}}
        )

        formatted_response = str(format_response(result.content))
        logger.info(f"格式化的搜索结果: {formatted_response}")

        # 处理流式响应
        if request.stream:
            # 定义一个异步生成器函数，用于生成流式数据
            async def generate_stream():
                # 为每个流式数据片段生成一个唯一的chunk_id
                chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
                # 将格式化后的响应按行分割
                lines = formatted_response.split('\n')
                # 历每一行，并构建响应片段
                for i, line in enumerate(lines):
                    # 创建一个字典，表示流式数据的一个片段
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        # "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": line + '\n'}, # if i > 0 else {"role": "assistant", "content": ""},
                                "finish_reason": None
                            }
                        ]
                    }
                    # 将片段转换为JSON格式并生成
                    yield f"{json.dumps(chunk)}\n"
                    # 每次生成数据后，异步等待0.5秒
                    await asyncio.sleep(0.5)
                # 生成最后一个片段，表示流式响应的结束
                final_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"{json.dumps(final_chunk)}\n"

            # 返回fastapi.responses中StreamingResponse对象，流式传输数据
            # media_type设置为text/event-stream以符合SSE(Server-SentEvents) 格式
            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        # 处理非流式响应处理
        else:
            response = ChatCompletionResponse(
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=Message(role="assistant", content=formatted_response),
                        finish_reason="stop"
                    )
                ]
            )
            logger.info(f"发送响应内容: {response}")
            # 返回fastapi.responses中JSONResponse对象
            # model_dump()方法通常用于将Pydantic模型实例的内容转换为一个标准的Python字典，以便进行序列化
            return JSONResponse(content=response.model_dump())

    except Exception as e:
        print(e)
        logger.error(f"处理聊天完成时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/v1/embeddings")
async def file_embedding(file: UploadFile = File(...), userId: str = Form(...)):
    if not model or not embeddings or not vectorstore or not prompt or not chain:
        logger.error("服务未初始化")
        raise HTTPException(status_code=500, detail="服务未初始化")
    
    try:
        logger.info(f"已接受文件{file.filename}")
        directory = "./uploads/"  
        os.makedirs(directory, exist_ok=True)

        # 获取不包含路径的文件名
        filename = os.path.basename(file.filename)

        # 构建文件的保存路径
        file_path = os.path.join(directory, filename)
        # logger.info(f"文件保存路径: {file_path}")

        # 将上传的文件保存到指定路径
        with open(file_path, "wb+") as file_object:
            file_object.write(await file.read())
            # logger.info(f"文件已保存到{file_path}")

        # # 写入文件  
        # with open(filepath, 'wb') as file_object:  # 使用 'wb' 模式
        #     file_object.write(await file.read()) 
            
            
        # 调用数据库存储向量
        embed_result = connetDB.vectorStoreSave(file_path, vector_db, userId)
        logger.info(f"已向量化文件{file_path}")
        
        # 返回向量化结果
        logger.info(embed_result)
        return {"status": "success", "message": "文件上传成功"}

    except Exception as e:
        logger.error(f"处理文件嵌入时出错:\n\n {str(e)}")
        return {"status": "error", "message": "{str(e)}"}

        

if __name__ == "__main__":
    logger.info(f"在端口 {PORT} 上启动服务器")
    # uvicorn是一个用于运行ASGI应用的轻量级、超快速的ASGI服务器实现
    # 用于部署基于FastAPI框架的异步PythonWeb应用程序
    uvicorn.run(app, host=HOST, port=PORT)


