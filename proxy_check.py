import os

import requests


url = "https://deepmimo.net/"
print("HTTP_PROXY :", os.environ.get("HTTP_PROXY", "<未设置>"))
print("HTTPS_PROXY:", os.environ.get("HTTPS_PROXY", "<未设置>"))
print("requests 使用的代理:", requests.utils.get_environ_proxies(url))

try:
    response = requests.get(url, timeout=30)
    print("DeepMIMO 官网响应:", response.status_code)
except requests.RequestException as error:
    print("DeepMIMO 官网连接失败:", repr(error))
    raise SystemExit(1)
