import pandas as pd
import logging
import re
from langchain_openai import OpenAIEmbeddings
from tools.getConfig import GetConfig


# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = GetConfig()  
config.initialize()


# 当处理中文文本时，按照标点进行断句
def sent_tokenize(input_string):
    sentences = re.split(r'(?<=[。！？；?!])', input_string)
    # 去掉空字符串
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def preprocess_comments(file_path):
    """
    从Excel文件中读取商品评论并进行预处理。
    :param file_path: Excel文件路径
    :return: 预处理后的评论列表
    """
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 确保评论列存在
    if '评论' not in df.columns or '用户名' not in df.columns or '星级' not in df.columns:
        logger.error("Excel文件中缺少必要的列：评论、用户名、星级")
        return []
    
    # 初始化预处理后的评论列表
    preprocessed_comments = []

    # 遍历每一条评论
    for index, row in df.iterrows():
        comment = row['评论']
        username = row['用户名']
        star_rating = row['星级']
        
        # 分句处理
        sentences = sent_tokenize(comment)
        
        # 进一步清理句子中的特殊字符
        cleaned_sentences = [re.sub(r'[^\w\s]', '', sentence) for sentence in sentences]
        
        # 将处理后的句子与用户名和星级信息整合
        processed_sentences = [
            f"{username} (星级: {star_rating}) - {sentence}" for sentence in cleaned_sentences
        ]
        
        # 将处理后的句子添加到结果列表
        preprocessed_comments.extend(processed_sentences)

    return preprocessed_comments

def batch_embed_texts(texts, batch_size=20):
    """
    批量嵌入文本。
    :param texts: 文本列表
    :param batch_size: 每批处理的文本数量
    :return: 向量化后的结果
    """
    embeddings = OpenAIEmbeddings(openai_api_base = config.openai['openai_api_base'],
                                openai_api_key = config.openai['openai_api_key'],
                                model = config.openai['embedding_model'])
    all_embeddings = []

    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings
    except Exception as e:
        logger.error(f"向量化时发生错误：{e}")
        return []

# 示例调用
if __name__ == "__main__":
    file_path = "C:/Users/admin/Desktop/test.xlsx"  # 替换为实际的文件路径
    preprocessed_comments = preprocess_comments(file_path)
    logger.info(f"预处理完成，共处理了 {len(preprocessed_comments)} 条句子。")

    # 批量向量化
    batch_size = 10
    embeddings = batch_embed_texts(preprocessed_comments, batch_size=batch_size)
    logger.info(f"向量化完成，共处理了 {len(embeddings)} 条向量。")

    # for comment in preprocessed_comments[:5]:
    #     logger.info(comment)

    # 打印前几条预处理后的评论及其向量
    for i in range(min(5, len(preprocessed_comments))):
        logger.info(f"Comment: {preprocessed_comments[i]}")
        logger.info(f"Embedding: {embeddings[i]}")