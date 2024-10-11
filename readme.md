# 基于Langchain+RAG的简易评论分析机器人
## 项目概述
个人练手项目，基于LangChain框架开发了一个前后端分离的智能问答系统，用户可以通过FastAPI提供的RESTful API与系统进行交互。系统集成了聊天功能、文件嵌入以及会话历史管理，可作为RAG项目基础项目使用，进行更多功能的开发。

## 主要功能
+ **聊天完成**：支持多轮对话，能够根据上下文理解用户意图并给出恰当的回答。
+ **文件嵌入**：允许用户上传文档，系统将自动提取文档中的信息用于增强知识库。
+ **会话历史管理**：记录用户的对话历史，以便更好地理解和回应用户的后续提问。
+ **配置文件与参数解析**：提供了灵活的配置选项，可以根据不同的应用场景调整系统行为。
+ **数据库处理**：集成简易的数据库操作功能。

## 技术栈
+ **前端框架**：Gradio
+ **后端框架**：FastAPI
+ **AI框架**：LangChain
+ **数据库**：Chroma, sqlite


## 安装与运行
1. 克隆仓库到本地：

```plain
bash

git clone <项目仓库地址>
cd 项目目录
```

2. 安装依赖：

```plain
bash

pip install -r requirements.txt
```

3. 配置环境变量（如数据库连接信息等）：

```plain
bash

cp .env.example .env
# 编辑.env文件以匹配您的环境设置
```

4. 启动服务：

```plain
bash

uvicorn app:app --reload
```

5. 访问API接口：
    - 文档：`http://localhost:8000/docs`
    - 健康检查：`http://localhost:8000/healthcheck`

## 感谢


