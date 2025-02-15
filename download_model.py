#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-0.5B', cache_dir="./weights")