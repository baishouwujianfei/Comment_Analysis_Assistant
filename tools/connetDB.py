# 功能说明：将PDF文件进行向量计算并持久化存储到向量数据库（chroma）

# 相关依赖库
# pip install openai chromadb
import sys
import os

# 获取当前脚本所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
root_dir = os.path.dirname(current_dir)

# 将项目根目录添加到 sys.path
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 引入相关库
import logging
import chromadb
import uuid
import numpy as np
from tools import excelSplit
from tools.getConfig import GetConfig

from langchain_community.vectorstores import Chroma



# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 模型设置相关  根据自己的实际情况进行调整
# API_TYPE = "oneapi"  # openai:调用gpt模型

config = GetConfig()  
config.initialize()

# openai模型相关配置 根据自己的实际情况进行调整
OPENAI_API_BASE = config.openai['openai_api_base']
OPENAI_EMBEDDING_API_KEY = config.openai['openai_api_key']
OPENAI_EMBEDDING_MODEL = config.openai['embedding_model']


# 设置测试文本类型
# 测试的文件路径
INPUT_FILE = r"C:/Users/admin/Desktop/test.xlsx"

# 指定文件中待处理的页码，全部页码则填None
PAGE_NUMBERS=None
# PAGE_NUMBERS=[2, 3]

# 指定向量数据库chromaDB的存储位置和集合 根据自己的实际情况进行调整
CHROMADB_DIRECTORY = config.chromaDB['directory']  # chromaDB向量数据库的持久化路径
CHROMADB_COLLECTION_NAME = config.chromaDB['collection_name']  # 待查询的chromaDB向量数据库的集合名称
# CHROMADB_COLLECTION_NAME = "demo002"  # 待查询的chromaDB向量数据库的集合名称

class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        # 申明使用全局变量
        global CHROMADB_DIRECTORY
        # 实例化一个chromadb对象
        # 设置一个文件夹进行向量数据库的持久化存储  路径为当前文件夹下chromaDB文件夹
        chroma_client = chromadb.PersistentClient(path=CHROMADB_DIRECTORY)
        # 创建一个collection数据集合
        # get_or_create_collection()获取一个现有的向量集合，如果该集合不存在，则创建一个新的集合
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name)
        # embedding处理函数
        self.embedding_fn = embedding_fn
        

    # 添加文档到集合
    # 文档通常包括文本数据和其对应的向量表示，这些向量可以用于后续的搜索和相似度计算
    def add_documents(self, documents, user_id):
        ids=[str(uuid.uuid4()) for i in range(len(documents))]
        metadatas = [{'user_id': user_id} for _ in range(len(documents))]  # 每个文档都附加用户ID作为元数据
        self.collection.add(
            embeddings=self.embedding_fn(documents),  # 调用函数计算出文档中文本数据对应的向量
            documents=documents,  # 文档的文本数据
            ids=ids,  # 文档的唯一标识符 自动生成uuid,128位  
            metadatas=metadatas  # 元数据，这里存储了用户ID
        )
        return self.collection.get(ids=ids)

    
    # 删除指定id文档
    def delete_documents(self, ids):
        self.collection.delete(ids=ids)
        
    # 检索向量数据库，返回包含查询结果的对象或列表，这些结果包括最相似的向量及其相关信息
    # query：查询文本
    # top_n：返回与查询向量最相似的前 n 个向量
    def search(self, query, top_n, user_id):
        try:
            results = self.collection.query(
                # 计算查询文本的向量，然后将查询文本生成的向量在向量数据库中进行相似度检索
                query_embeddings=self.embedding_fn([query]),
                n_results=top_n,
                where={"user_id": user_id}
            )
            return results
        except Exception as e:
            logger.info(f"检索向量数据库时出错: {e}")
            return []
        
# 封装文本预处理及灌库方法  提供外部调用
def vectorStoreSave(input_file, vector_db, user_id):
    try:
        global CHROMADB_COLLECTION_NAME, INPUT_FILE, PAGE_NUMBERS
        INPUT_FILE = input_file
       
        # 1、获取处理后的文本数据
        # 演示测试对指定的全部页进行处理，其返回值为划分为段落的文本列表
        paragraphs = excelSplit.preprocess_comments(file_path=INPUT_FILE)
        # 2、将文本片段灌入向量数据库
        # 实例化一个向量数据库对象
        # 其中，传参collection_name为集合名称, embedding_fn为向量处理函数
        # 向向量数据库中添加文档（文本数据、文本数据对应的向量数据）
        vector_db.add_documents(paragraphs, user_id)
        logger.info(f"文本灌库成功，向量数据库中添加了{input_file}文档")
        return {"message": f"文本灌库成功"}
    except Exception as e:
        logger.error(f"文本灌库失败{e}")
        return {"message": f"文本灌库失败{e}"}


def vectorStoreSearch(query_text, vector_db, user_id):
    """
    测试向量数据库查询功能的函数。
    
    :param query_text: 用于查询的文本字符串
    """
    # 执行查询操作
    results = vector_db.search(query_text, top_n=3, user_id=user_id)
    
    logger.info(f"检索向量数据库的结果: {results}")


if __name__ == "__main__":
    # 测试文本预处理及灌库
    vector_db = MyVectorDBConnector(CHROMADB_COLLECTION_NAME, excelSplit.batch_embed_texts)
    # save1 = vectorStoreSave(input_file=INPUT_FILE, vector_db=vector_db, user_id="user1111111")
    # ids = save1["ids"]
    # vector_db.delete_documents(ids)
    vectorStoreSearch(query_text="这个灯使用的时候老是闪烁", vector_db=vector_db, user_id="user1111112")
