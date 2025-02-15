# 下载数据集
from modelscope.msdatasets import MsDataset

ds =  MsDataset.load('testUser/GSM8K_zh', subset_name='default', split='train', cache_dir="./gsm8k", trust_remote_code=True)