import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder


api_url = "http://localhost:8012/v1/embeddings"

# 用户ID
user_id = "uuu444"

# 要上传的文件路径
file_path = "C:/Users/admin/Desktop/test.xlsx"

with open(file_path, 'rb') as f:
    # 构建请求中的文件部分，'file'是接口中定义的字段名
    files = {'file': (file_path, f)}
    # 构建请求中的表单部分，'userId'是接口中定义的字段名
    data = {'userId': user_id}
    # 发送请求
    response = requests.post(api_url, files=files, data=data)

# 打印响应状态码和内容
print(f"Response Status Code: {response.status_code}")
print(f"Response Content: {response.text}")