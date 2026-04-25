# 本地大模型推理框架详细指南

## 目录
1. [Ollama](#1-ollama)
2. [llama.cpp / llama-cpp-python](#2-llamacpp--llama-cpp-python)
3. [vLLM](#3-vllm)
4. [HuggingFace Transformers](#4-huggingface-transformers)
5. [LM Studio](#5-lm-studio)
6. [Xinference](#6-xinference)
7. [Text Generation Inference (TGI)](#7-text-generation-inference-tgi)

---

## 1. Ollama

### 1.1 简介
Ollama 是一个开源的大语言模型本地运行工具，旨在让用户能够在本地设备上轻松运行大型语言模型。它采用了类似 Docker 的设计理念，通过简单的命令即可拉取和运行各种模型。

### 1.2 核心特性
- **一键安装**：提供 Windows、macOS、Linux 安装包
- **模型管理**：类似 Docker 的 pull/run 机制
- **REST API**：内置 API 服务，端口 11434
- **多模型支持**：支持 Llama 3、Qwen、Mistral 等主流模型
- **Modelfile**：支持自定义模型配置

### 1.3 安装方式

**Windows:**
```bash
# 下载安装包
# 访问 https://ollama.com/download 下载 Windows 版本
# 或使用 winget
winget install Ollama.Ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

### 1.4 使用示例

**命令行运行:**
```bash
# 拉取模型
ollama pull qwen2.5:3b

# 运行模型
ollama run qwen2.5:3b

# 查看已安装模型
ollama list

# 删除模型
ollama rm qwen2.5:3b
```

**Python API:**
```python
import ollama

# 简单对话
response = ollama.chat(
    model='qwen2.5:3b',
    messages=[{'role': 'user', 'content': '你好'}]
)
print(response['message']['content'])

# 流式输出
stream = ollama.chat(
    model='qwen2.5:3b',
    messages=[{'role': 'user', 'content': '讲个故事'}],
    stream=True,
)
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```

**REST API:**
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:3b",
  "messages": [{"role": "user", "content": "你好"}]
}'
```

**自定义 Modelfile:**
```dockerfile
FROM qwen2.5:3b
# 设置系统提示
SYSTEM "你是一个专业的编程助手"
# 设置参数
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
```
```bash
ollama create my-assistant -f Modelfile
```

### 1.5 模型格式
- 主要支持 GGUF 格式
- 通过 Modelfile 可导入自定义模型

### 1.6 硬件要求
| 模型规模 | 最低内存 | 推荐配置 |
|---------|---------|---------|
| 3B | 4GB | CPU 4核 |
| 7B | 8GB | CPU 8核 |
| 13B | 16GB | CPU 8核 + GPU |
| 70B | 64GB | GPU 24GB+ |

### 1.7 优缺点
**优点:**
- 安装使用最简单
- 跨平台支持好
- 社区活跃，模型丰富
- 内置 API 服务

**缺点:**
- 推理性能不是最优
- 不支持高级推理参数调优
- 模型格式限制（主要 GGUF）

### 1.8 适用场景
- 个人开发者快速原型开发
- 本地测试和体验大模型
- 小型项目集成
- 离线环境使用

---

## 2. llama.cpp / llama-cpp-python

### 2.1 简介
llama.cpp 是一个用 C/C++ 编写的高效 LLM 推理引擎，由 Georgi Gerganov 开发。它专注于在消费级硬件上实现高性能推理，支持纯 CPU 推理和 GPU 加速。llama-cpp-python 是其 Python 绑定。

### 2.2 核心特性
- **纯 C/C++ 实现**：无外部依赖，编译即用
- **CPU 推理优化**：支持量化、多线程
- **GPU 加速**：支持 CUDA、Metal、Vulkan、SYCL
- **量化支持**：支持多种量化格式（Q4_0、Q5_0、Q8_0 等）
- **低内存占用**：可在普通笔记本上运行 7B 模型

### 2.3 安装方式

**llama.cpp (C++):**
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

**llama-cpp-python:**
```bash
# 仅 CPU
pip install llama-cpp-python

# 带 CUDA 支持
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# 带 Metal 支持 (macOS)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

### 2.4 使用示例

**命令行 (llama.cpp):**
```bash
# 基本推理
./build/bin/main -m ./model.gguf -p "你好" -n 128

# 交互式对话
./build/bin/main -m ./model.gguf -i -ins

# 指定 GPU 层数
./build/bin/main -m ./model.gguf -p "你好" -ngl 35 -n 128

# 多线程
./build/bin/main -m ./model.gguf -p "你好" -t 8 -n 128
```

**Python API:**
```python
from llama_cpp import Llama

# 初始化模型
llm = Llama(
    model_path="./qwen2.5-3b.Q4_K_M.gguf",
    n_ctx=4096,        # 上下文长度
    n_threads=8,       # CPU 线程数
    n_gpu_layers=30,   # GPU 卸载层数 (0=纯CPU)
    verbose=False
)

# 文本生成
output = llm(
    "Q: 什么是人工智能？\nA:",
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
    echo=False
)
print(output['choices'][0]['text'])

# 对话模式
messages = [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "你好"}
]
response = llm.create_chat_completion(messages=messages)
print(response['choices'][0]['message']['content'])

# 流式输出
stream = llm.create_chat_completion(messages=messages, stream=True)
for chunk in stream:
    delta = chunk['choices'][0]['delta']
    if 'content' in delta:
        print(delta['content'], end='', flush=True)
```

**创建 OpenAI 兼容服务器:**
```bash
# 启动服务器
python -m llama_cpp.server --model ./model.gguf --host 0.0.0.0 --port 8000

# 使用 OpenAI SDK 调用
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "你好"}]
)
```

### 2.5 模型格式
- **GGUF** (推荐)：最新格式，支持元数据和改进的量化
- GGML：旧格式，已弃用
- GPTQ：支持 4-bit 量化

### 2.6 量化格式对比
| 格式 | 大小 (7B) | 质量 | 速度 |
|------|----------|------|------|
| Q2_K | 2.6GB | 较低 | 最快 |
| Q4_K_M | 4.2GB | 良好 | 快 |
| Q5_K_M | 4.8GB | 较好 | 中 |
| Q8_0 | 7.2GB | 接近原始 | 慢 |
| F16 | 14GB | 原始 | 最慢 |

### 2.7 硬件要求
| 模型规模 | Q4_K_M 内存 | Q8_0 内存 |
|---------|------------|----------|
| 3B | 2GB | 3.5GB |
| 7B | 4GB | 7GB |
| 13B | 8GB | 13GB |
| 70B | 36GB | 70GB |

### 2.8 优缺点
**优点:**
- 资源占用极低
- 支持纯 CPU 推理
- 跨平台（Windows、macOS、Linux）
- 支持 Apple Silicon
- 量化选项丰富

**缺点:**
- 需要手动转换模型为 GGUF
- 多模态支持有限
- 分布式推理不支持
- 编译安装可能遇到问题

### 2.9 适用场景
- 资源受限环境（笔记本、边缘设备）
- 无 GPU 或只有集成显卡
- Apple Silicon Mac
- 需要离线部署的场景
- IoT 设备

---

## 3. vLLM

### 3.1 简介
vLLM 是由 UC Berkeley 开发的高性能 LLM 推理和服务库。它引入了 PagedAttention 技术，显著提高了推理吞吐量和内存效率，是目前最快的开源推理框架之一。

### 3.2 核心特性
- **PagedAttention**：创新的注意力算法，内存效率高
- **连续批处理**：动态处理不同长度的请求
- **高吞吐量**：比 HuggingFace Transformers 快 2-4 倍
- **多 GPU 支持**：张量并行、流水线并行
- **OpenAI 兼容 API**：无缝替换 OpenAI API
- **流式输出**：支持 token-by-token 流式传输

### 3.3 安装方式

**Linux (推荐):**
```bash
pip install vllm
```

**带特定 CUDA 版本:**
```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
```

**Docker:**
```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest
```

**Windows:**
```bash
# 需要启用长路径支持（管理员 PowerShell）
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
# 重启后
pip install vllm
```

### 3.4 使用示例

**Python API:**
```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="./qwen2.5-3b",
    tensor_parallel_size=1,    # GPU 数量
    gpu_memory_utilization=0.9, # GPU 内存利用率
    max_model_len=4096
)

# 批量生成
prompts = [
    "人工智能的未来是",
    "解释量子计算",
    "写一首关于春天的诗"
]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}\n")
```

**OpenAI 兼容服务器:**
```bash
# 启动服务
python -m vllm.entrypoints.openai.api_server \
    --model ./qwen2.5-3b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1

# 使用 curl 测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./qwen2.5-3b",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

**Python 客户端:**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")

# 聊天补全
response = client.chat.completions.create(
    model="./qwen2.5-3b",
    messages=[{"role": "user", "content": "你好"}],
    temperature=0.7,
    max_tokens=100
)
print(response.choices[0].message.content)

# 流式输出
stream = client.chat.completions.create(
    model="./qwen2.5-3b",
    messages=[{"role": "user", "content": "讲个故事"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
```

### 3.5 模型格式
- HuggingFace Transformers 格式（直接加载）
- 支持 Safetensors、PyTorch checkpoint

### 3.6 硬件要求
| 模型规模 | 最小显存 | 推荐显存 |
|---------|---------|---------|
| 3B | 4GB | 6GB |
| 7B | 8GB | 12GB |
| 13B | 16GB | 24GB |
| 70B | 80GB+ | 多 GPU |

### 3.7 优缺点
**优点:**
- 推理性能极佳
- 高并发处理能力强
- 支持多 GPU 分布式推理
- OpenAI API 兼容
- 连续批处理提高效率

**缺点:**
- 需要较大显存
- Windows 支持有限
- 安装可能遇到编译问题
- 不支持 CPU 推理

### 3.8 适用场景
- 生产环境 API 服务
- 高并发应用场景
- 多用户共享服务
- 需要最大吞吐量的场景
- 企业级部署

---

## 4. HuggingFace Transformers

### 4.1 简介
HuggingFace Transformers 是最流行的开源 NLP 库，提供了数千个预训练模型。它是最灵活的推理框架，支持几乎所有主流模型架构。

### 4.2 核心特性
- **模型丰富**：支持 100+ 架构，数千预训练模型
- **多模态**：支持文本、图像、音频、视频
- **生态完善**：与 HuggingFace Hub 无缝集成
- **灵活定制**：可修改模型架构、训练、推理全流程
- **多后端**：支持 PyTorch、TensorFlow、JAX

### 4.3 安装方式

**基础安装:**
```bash
pip install transformers
```

**带 PyTorch:**
```bash
pip install transformers torch
```

**完整安装:**
```bash
pip install transformers torch accelerate sentencepiece protobuf
```

**GPU 支持:**
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4.4 使用示例

**Pipeline (最简单):**
```python
from transformers import pipeline

# 文本生成
generator = pipeline("text-generation", model="./qwen2.5-3b", device=0)
result = generator("你好，请介绍一下自己", max_new_tokens=100)
print(result[0]['generated_text'])

# 对话
chatbot = pipeline(
    "text-generation",
    model="./qwen2.5-3b",
    device_map="auto",  # 自动选择设备
    torch_dtype="auto"
)
messages = [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "你好"}
]
# 需要手动格式化消息
```

**直接使用模型:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载模型和分词器
model_name = "./qwen2.5-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 推理
text = "人工智能的未来"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**对话模板:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("./qwen2.5-3b")
model = AutoModelForCausalLM.from_pretrained(
    "./qwen2.5-3b",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 构建对话
messages = [
    {"role": "system", "content": "你是一个专业的编程助手"},
    {"role": "user", "content": "如何用 Python 写一个快速排序？"}
]

# 应用聊天模板
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

**流式生成:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("./qwen2.5-3b")
model = AutoModelForCausalLM.from_pretrained(
    "./qwen2.5-3b",
    torch_dtype=torch.float16,
    device_map="auto"
)

text = "写一首诗"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 使用 TextStreamer 实现流式输出
from transformers import TextStreamer
streamer = TextStreamer(tokenizer, skip_prompt=True)

model.generate(
    **inputs,
    max_new_tokens=200,
    streamer=streamer
)
```

**量化推理 (bitsandbytes):**
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "./qwen2.5-3b",
    quantization_config=bnb_config,
    device_map="auto"
)
# 7B 模型显存占用从 14GB 降至约 4GB
```

### 4.5 模型格式
- PyTorch (pytorch_model.bin)
- Safetensors (model.safetensors)
- TensorFlow (tf_model.h5)
- GGUF (需要 llama-cpp-python 或 transformers[gguf])

### 4.6 硬件要求
| 模型规模 | FP16 显存 | 4-bit 显存 |
|---------|----------|-----------|
| 3B | 6GB | 2GB |
| 7B | 14GB | 4GB |
| 13B | 26GB | 8GB |
| 70B | 140GB | 40GB |

### 4.7 优缺点
**优点:**
- 模型生态最丰富
- 支持所有主流模型
- 文档完善，社区活跃
- 支持训练和微调
- 灵活度最高

**缺点:**
- 推理性能不如专用框架
- 需要较多显存
- 批量处理能力较弱
- 需要更多代码

### 4.8 适用场景
- 研究和开发
- 模型微调和训练
- 需要自定义模型行为
- 多模态任务
- 学术项目

---

## 5. LM Studio

### 5.1 简介
LM Studio 是一个图形化的大模型本地运行工具，提供了完整的模型管理、下载、运行和 API 服务。它是最适合非技术用户的解决方案。

### 5.2 核心特性
- **图形界面**：完全可视化操作
- **内置模型搜索**：直接从 HuggingFace 搜索下载
- **本地 API 服务**：OpenAI 兼容 API
- **模型比较**：同时运行多个模型对比
- **无需命令行**：零代码使用

### 5.3 安装方式
```
# 访问官网下载
# https://lmstudio.ai/

# 支持 Windows、macOS、Linux
```

### 5.4 使用示例

**图形界面操作:**
1. 打开 LM Studio
2. 点击 "Search" 搜索模型
3. 选择模型点击下载
4. 下载完成后点击 "Chat" 开始对话
5. 调整参数（温度、上下文长度等）

**启动本地 API:**
1. 点击左侧 "Local Server" 标签
2. 选择要加载的模型
3. 点击 "Start Server"
4. API 地址：`http://localhost:1234/v1`

**使用 API:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"  # 任意字符串
)

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "你好"}],
    temperature=0.7,
    max_tokens=100
)
print(response.choices[0].message.content)
```

### 5.5 模型格式
- GGUF（主要支持）
- 自动从 HuggingFace 下载

### 5.6 硬件要求
与 llama.cpp 类似（底层使用 llama.cpp）

### 5.7 优缺点
**优点:**
- 零学习成本
- 图形界面直观
- 内置模型管理
- 自动处理依赖
- 适合非技术人员

**缺点:**
- 闭源软件
- 功能更新依赖官方
- 性能不如命令行工具
- 自定义选项有限

### 5.8 适用场景
- 非技术用户使用
- 快速体验大模型
- 教育和演示
- 不想使用命令行的用户

---

## 6. Xinference

### 6.1 简介
Xinference 是一个开源的模型推理服务平台，由 Xorbits 团队开发。它支持多种模型类型和框架，提供了统一的 API 接口和管理界面。

### 6.2 核心特性
- **多模型支持**：LLM、Embedding、Rerank、Image
- **多框架后端**：vLLM、llama.cpp、Transformers
- **集群部署**：支持分布式部署
- **模型管理**：Web UI 管理模型
- **OpenAI 兼容**：统一 API 接口

### 6.3 安装方式

**pip 安装:**
```bash
pip install "xinference[all]"
```

**Docker:**
```bash
docker run -p 9997:9997 -e XINFERENCE_HOME=/data -v /data:/data xprobe/xinference:latest
```

### 6.4 使用示例

**启动服务:**
```bash
# 启动单节点
xinference

# 启动集群
xinference-supervisor
xinference-worker --supervisor-addr <supervisor-ip>:9997
```

**Web UI:**
```
# 访问 http://localhost:9997
# 在界面中搜索、下载、启动模型
```

**命令行:**
```bash
# 启动模型
xinference launch --model-name qwen2.5 --size-in-billions 3 --model-format ggufv2

# 列出运行中的模型
xinference list

# 停止模型
xinference terminate --model-uid <model-uid>
```

**Python API:**
```python
from xinference.client import RESTfulClient

client = RESTfulClient("http://localhost:9997")

# 启动模型
model_uid = client.launch_model(
    model_name="qwen2.5",
    model_size_in_billions=3,
    model_format="ggufv2",
    quantization="Q4_K_M"
)

# 推理
model = client.get_model(model_uid)
response = model.chat(
    messages=[{"role": "user", "content": "你好"}],
    generate_config={"max_tokens": 100}
)
print(response['choices'][0]['message']['content'])
```

**OpenAI 兼容 API:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9997/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model=model_uid,
    messages=[{"role": "user", "content": "你好"}]
)
```

### 6.5 模型格式
- GGUF
- HuggingFace Transformers
- PyTorch

### 6.6 优缺点
**优点:**
- 支持多种模型类型
- 多后端自动选择
- Web UI 管理方便
- 支持集群部署
- 适合团队共享

**缺点:**
- 相对较新，生态不如其他成熟
- 文档还在完善中
- 某些高级功能需要学习

### 6.7 适用场景
- 团队共享模型服务
- 需要多种模型类型（LLM、Embedding 等）
- 中小型企业的模型服务
- 需要统一管理界面

---

## 7. Text Generation Inference (TGI)

### 7.1 简介
TGI 是 HuggingFace 开发的生产级文本生成推理服务器。它被用于 HuggingChat 和 HuggingFace 的推理 API，是一个经过生产验证的高性能框架。

### 7.2 核心特性
- **生产级稳定性**：经过 HuggingFace 生产验证
- **张量并行**：支持多 GPU 分布式推理
- **连续批处理**：类似 vLLM 的优化
- **流式输出**：支持 token 级别流式传输
- **指标监控**：内置 Prometheus 指标
- **安全机制**：支持 token 验证

### 7.3 安装方式

**Docker (推荐):**
```bash
# 单 GPU
docker run --gpus all -p 8080:80 \
    -v /data:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id ./model_path \
    --sharded false

# 多 GPU (张量并行)
docker run --gpus all -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id ./model_path \
    --num-shard 4
```

**源码编译 (Linux):**
```bash
git clone https://github.com/huggingface/text-generation-inference
cd text-generation-inference
make install
```

### 7.4 使用示例

**启动服务:**
```bash
docker run --gpus all -p 8080:80 \
    -v /data:/data \
    ghcr.io/huggingface/text-generation-inference:2.0 \
    --model-id Qwen/Qwen2.5-3B \
    --max-input-length 4096 \
    --max-total-tokens 8192 \
    --max-batch-prefill-tokens 4096 \
    --max-batch-total-tokens 8192
```

**使用 API:**
```bash
# curl 测试
curl 127.0.0.1:8080/generate \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{
        "inputs": "你好",
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }'
```

**Python 客户端:**
```python
from text_generation import Client

client = Client("http://127.0.0.1:8080")

# 生成
response = client.generate("你好", max_new_tokens=100)
print(response.generated_text)

# 流式生成
for response in client.generate_stream("你好", max_new_tokens=100):
    if not response.token.special:
        print(response.token.text, end='', flush=True)
```

**OpenAI 兼容 (TGI 2.0+):**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="tgi",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=100
)
```

### 7.5 模型格式
- HuggingFace Transformers
- Safetensors
- PyTorch

### 7.6 硬件要求
与 vLLM 类似，需要较大显存

### 7.7 优缺点
**优点:**
- 生产级稳定性
- HuggingFace 官方支持
- 多模型架构优化
- 内置监控和日志
- 安全机制完善

**缺点:**
- 仅支持 Linux/Docker
- 配置参数复杂
- 需要较大显存
- 不支持 Windows

### 7.8 适用场景
- 企业级生产部署
- 需要高可用性
- 需要监控和日志
- HuggingFace 生态集成
- 大规模服务

---

## 综合对比表

| 特性 | Ollama | llama.cpp | vLLM | Transformers | LM Studio | Xinference | TGI |
|------|--------|-----------|------|-------------|-----------|------------|-----|
| 安装难度 | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ |
| 推理性能 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| CPU 支持 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| GPU 支持 | 有限 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 多 GPU | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Windows | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ❌ |
| macOS | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| Linux | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| OpenAI API | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| 图形界面 | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| 模型格式 | GGUF | GGUF | HF | HF | GGUF | 多格式 | HF |
| 适合场景 | 个人 | 边缘 | 生产 | 研究 | 个人 | 团队 | 企业 |

## 选择建议

### 个人开发者 / 快速原型
```
推荐：Ollama 或 LM Studio
理由：安装简单，开箱即用
```

### 资源受限环境
```
推荐：llama-cpp-python
理由：支持 CPU 推理，量化选项丰富
```

### 高性能生产服务
```
推荐：vLLM 或 TGI
理由：吞吐量高，稳定性好
```

### 研究和开发
```
推荐：HuggingFace Transformers
理由：灵活度高，生态完善
```

### 团队共享服务
```
推荐：Xinference
理由：管理方便，支持多模型类型
```

### 企业级部署
```
推荐：TGI 或 vLLM + Docker/K8s
理由：生产验证，可监控，可扩展