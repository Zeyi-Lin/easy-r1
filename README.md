# easy r1

[SwanLab训练过程](https://swanlab.cn/@ZeyiLin/Qwen-R1-Zero/charts)

## 1. 环境准备

在运行下面的命令之前，请先保证你安装了Python3.10及以上版本，计算机中有Nvidia显卡，并且安装了CUDA和cuDNN。

```bash
pip install -r requirements.txt
```

## 2. 模型与数据集下载

- 模型：[Qwen2.5-0.5B](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B/summary)（注意是Base模型，而非Instruct模型）
- 数据集：[GSM8K_zh](https://modelscope.cn/datasets/testUser/GSM8K_zh/summary)

```bash
python download_model.py
python download_data.py
```

## 3. 开启训练

在开启训练之前，如果你还没有使用过[SwanLab](https://swanlab.cn/)，请先注册一个账号，登录后在[设置](https://swanlab.cn/settings)页面复制你的API Key，然后执行：

```bash
swanlab login
```

将你的API Key粘贴进去，然后按回车完成登录。

> ps：如果你对命令行登录不习惯，也可以使用`swanlab.login()`函数进行登录，[指引](https://docs.swanlab.cn/api/py-login.html)
---

然后就可以开始训练了：

```bash
python train.py
```

