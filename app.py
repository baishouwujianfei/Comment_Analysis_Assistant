import gradio as gr
import requests
from requests.exceptions import HTTPError  
import json
from tools.getConfig import GetConfig
from tools.argument_parser import ArgumentParser
import time
import uuid
import logging

argument_parser = ArgumentParser()
args = argument_parser.parse_arguments()
config = GetConfig()  
config.initialize(args)

# openai模型相关配置 根据自己的实际情况进行调整
CONNECT_SERVER = config.api_service['host']
CONNECT_SERVER_PORT = config.api_service['port']

MAX_HISTROY_COUNT = 0


url = f"http://{CONNECT_SERVER}:{CONNECT_SERVER_PORT}"
headers = {"Content-Type": "application/json"}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [Line: %(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)


# 设置最大历史记录条数
def set_max_histroy_count(count):
    global MAX_HISTROY_COUNT
    MAX_HISTROY_COUNT = count
    # print(f"当前滑动设置{count}")


# 聊天函数
def slow_echo(message, history, request: gr.Request):
    global MAX_HISTROY_COUNT
    num = MAX_HISTROY_COUNT
    # request请求/v1/chat/completions接口进行聊天
    
    # 如果history元素大于num，则删除前num个元素
    while len(history) > num:
        history.pop(0)
    print(f"当前历史条数{len(history)}")
    print(f"当前设置值{num}")
    print(history)

    data = {
    "messages": [{"role": "user", "content": message}],
    "stream": True,
    "userId": request.session_hash,
    "conversationId":"456",
    "history_limit":num
    }

    try:
        with requests.post(url=url+"/v1/chat/completions", stream=True, headers=headers, data=json.dumps(data)) as response:
            content = ''
            for line in response.iter_lines():
                
                if line:
                    json_str = line.decode('utf-8').strip("data: ")
                    # 检查是否为空或不合法的字符串
                    if not json_str:
                        
                        continue
                    # 确保字符串是有效的JSON格式
                    if json_str.startswith('{') and json_str.endswith('}'):
                        try:
                            data = json.loads(json_str)
                            if data['choices'][0]['finish_reason'] == "stop":
                                logger.info(f"接收JSON数据结束")
                            else:
                                logger.info(f"流式输出，响应内容是: {data['choices'][0]['delta']['content']}")
                                for char in data['choices'][0]['delta']['content']:
                                    content += char
                                    time.sleep(0.01)
                                    yield content

                        except json.JSONDecodeError as e:
                            logger.info(f"JSON解析错误: {e}")
                    else:
                        print(f"无效JSON格式: {json_str}")

    except Exception as e:
        print(f"Error occurred: {e}")
    
    
    # for i in range(len(message)):
    #     time.sleep(0.001)
        
    #     yield "You typed: " + message[: i + 1]


# 文件上传函数
def file_embedding(file_path, request: gr.Request): 
    with open(file_path, 'rb') as f:
        # 构建请求中的文件部分，'file'是接口中定义的字段名
        files = {'file': (file_path, f)}
        # 构建请求中的表单部分，'userId'是接口中定义的字段名
        data = {'userId': request.session_hash}
        
        # 发送请求
        try:
            response = requests.post(url+"/v1/embeddings", files=files, data=data) 
            logger.info(f"已上传文件: {file_path}")
        
        except HTTPError as http_err:  
            logger.error(f"HTTP error occurred: {http_err}")  
            logger.error(f"Response content: {response.text}")  # 打印响应内容以获取更多信息  
        except Exception as err:  
            logger.error(f"An error occurred: {err}")  
            raise

def print_filename(filename):
    print(filename)
    return "filename: " + filename

with gr.ChatInterface(  # theme=gr.themes.Monochrome(),
                        title = "chat with excel",
                        fn=slow_echo, 
                        submit_btn = "提交",
                        stop_btn = "停止",
                        retry_btn = "🔄  重试",
                        undo_btn = "↩️ 回退",
                        clear_btn = "🗑️  清除",
                        # examples = ["你好", "你好啊", "你好呀"],
                        autofocus=True,
                        # fill_width= True,
                        examples_per_page = 2
                        ) as demo: 
    with gr.Row(equal_height=False):
        # 最大历史设置条
        max_histroy = gr.Slider(
            label="最大记录对话轮数",
            info="请输入一个整数，表示最大记录对话轮数，默认为0",
            # value=5,
            scale=1,
            step=1,
            maximum=20,
        )
        max_histroy.change(fn=set_max_histroy_count, inputs=[max_histroy], outputs=None)
        # 文件上传按钮
        input_btn = gr.File(label="上传excel文件", file_count="single", file_types=["xlsx", "xls"], scale=2)
        input_btn.upload(fn=file_embedding, inputs=[input_btn], outputs=gr.Info('已上传文件'))

if __name__ == "__main__":
    demo.queue(200)  # 请求队列
    demo.launch(server_name='0.0.0.0',max_threads=500) # 线程池
    