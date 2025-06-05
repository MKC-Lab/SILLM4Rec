# SILLM4Rec: Sequential Item Recommendation with Large Language Models

本项目实现了基于大语言模型的序列化商品推荐系统，支持监督微调(SFT)和直接偏好优化(DPO)两种训练方式。

## 🚀 快速开始

### 1. 环境设置

#### 1.1 系统要求

- **Python**: 3.10
- **CUDA**: 12.1
- **GPU**: 建议使用带有大显存的 GPU（例如A100 40G）

#### 1.2 安装依赖

**重要**: 请按照以下顺序安装依赖以确保兼容性。

首先安装 PyTorch GPU 版本 (CUDA 12.1)：

```bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

然后安装其他依赖：

```bash
pip install -r requirements.txt
```

验证 CUDA 是否可用：

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
```

### 2. 数据准备

#### 2.1 下载原始数据

前往 [Amazon Review Data 2023](https://amazon-reviews-2023.github.io/) 网站下载以下文件(5 core)：

- **元数据文件**: `meta_[CATEGORY].jsonl` (如 `meta_Baby_Products.jsonl`)
- **评论数据文件**: `[CATEGORY].jsonl` (如 `Baby_Products.jsonl`)  
- **交互数据文件**: `[CATEGORY].test.csv` (如 `Baby_Products.test.csv`)

将下载的文件放置在 `raw_data/` 目录下。

#### 2.2 数据预处理

使用 `raw_data_process.ipynb` 处理原始数据：

1. 打开 `raw_data_process.ipynb`
2. 根据您的数据类别修改文件路径
3. 运行所有单元格生成处理后的数据

预处理步骤包括：

- 生成用户和物品的映射关系
- 转换 JSONL 格式为 Parquet 格式
- 筛选高质量的用户交互数据
- 生成训练集和测试集

### 3. 训练数据生成

使用 `training_data_process.ipynb` 生成训练数据和评估结果：

1. 配置 API 密钥和基础 URL（用于调用大语言模型）
2. 运行图像描述生成（将商品图片转换为文本描述）
3. 生成用户偏好摘要
4. 创建候选商品排序任务
5. 生成 SFT 和 DPO 训练数据

主要功能：

- **图像到文本转换**: 使用视觉语言模型描述商品图片
- **用户偏好分析**: 基于历史交互生成用户偏好摘要
- **排序任务生成**: 创建商品排序任务和对应的标准答案
- **训练数据构建**: 生成适用于不同训练方法的数据格式

### 4. 模型准备

#### 4.1 下载基础模型

在开始训练前，需要先下载并准备基础模型：

```bash
# 下载模型到本地目录，例如：
# models/DeepSeek-R1-Distill-Qwen-7B/
# 或使用 Hugging Face Hub 下载
```

确保模型文件完整并放置在适当的目录中。

### 5. 模型训练

#### 5.1 监督微调 (SFT)

使用 `SFT.ipynb` 进行监督微调：

1. 在 notebook 中配置基础模型路径
2. 加载预训练模型（如 DeepSeek-R1-Distill-Qwen-7B）
3. 配置 LoRA 参数
4. 加载 SFT 训练数据
5. 开始训练

特性：

- 支持 4bit 量化加载
- 使用 LoRA 高效微调
- 自动梯度检查点
- 支持多种优化器

#### 5.2 直接偏好优化 (DPO)

使用 `DPO.ipynb` 进行偏好优化：

1. 加载 SFT 微调后的模型
2. 配置 DPO 训练参数
3. 加载 DPO 训练数据（包含 chosen 和 rejected 样本）
4. 开始 DPO 训练

特性：

- 基于 SFT 模型进行进一步优化
- 自动处理偏好对数据
- 支持自定义 β 参数
- 内存效率优化

### 6. 模型部署

#### 6.1 安装 vLLM

训练完成后，推荐使用 vLLM 进行高效推理部署：

```bash
pip install vllm
```

#### 6.2 启动推理服务

使用以下命令启动 vLLM 推理服务：

```bash
vllm serve {模型位置} --max_model_len=4096 --override-generation-config "{\"temperature\": 0.2}"
```

示例：

```bash
# 部署 SFT 模型
vllm serve ./new_model/qwen-sft --max_model_len=4096 --override-generation-config "{\"temperature\": 0.2}"

# 部署 DPO 模型
vllm serve ./new_model/qwen-dpo --max_model_len=4096 --override-generation-config "{\"temperature\": 0.2}"
```

#### 6.3 使用推理服务

vLLM 服务启动后，可以通过 HTTP API 进行推理调用：

```python
api_key = "EMPTY"
base_url = "http://localhost:8000/v1"
```

## 📁 项目结构

```text
SILLM4Rec/
├── raw_data/                    # 原始数据目录
│   ├── meta_[CATEGORY].jsonl    # 商品元数据
│   ├── [CATEGORY].jsonl         # 用户评论数据
│   └── [CATEGORY].test.csv      # 交互测试数据
├── [category]_data/             # 处理后的数据目录
│   ├── train/                   # 训练数据
│   └── test/                    # 测试数据
├── raw_data_process.ipynb       # 数据预处理
├── training_data_process.ipynb  # 训练数据生成
├── SFT.ipynb                    # 监督微调
├── DPO.ipynb                    # 直接偏好优化
├── matrix.py                    # 评估指标计算
├── requirements.txt             # 依赖包列表
└── README.md                    # 项目说明
```

## 🔧 配置说明

### API 配置

在 `training_data_process.ipynb` 中配置您的 LLM API：

```python
api_key = "your_api_key_here"
base_url = "your_base_url_here"
```

### 模型路径配置

根据您的模型存放位置修改各个 notebook 中的路径：

```python
# SFT.ipynb 中 - 基础模型路径
model_name = "/path/to/your/base/model"  # 例如: "./models/DeepSeek-R1-Distill-Qwen-7B"

# DPO.ipynb 中 - SFT 微调后的模型路径  
model_name = "/path/to/your/sft/model"   # 例如: "./new_model/qwen-sft"

# 训练输出路径
output_dir = "/path/to/output"           # 例如: "./outputs"
save_dir = "/path/to/save/model"        # 例如: "./new_model/qwen-sft"
```

### 目录结构建议

```text
SILLM4Rec/
├── models/                      # 基础模型目录
│   └── DeepSeek-R1-Distill-Qwen-7B/
├── new_model/                   # 微调后模型目录
│   ├── qwen-sft/               # SFT 模型
│   └── qwen-dpo/               # DPO 模型
├── outputs/                     # 训练输出目录
└── ...                         # 其他项目文件
```

## 📊 评估指标

项目支持以下评估指标：

- **NDCG@K**: 归一化折扣累积增益
- **Valid Rate**: 有效推荐率

## 🛠️ 故障排除

### 环境相关问题

1. **Python 版本不兼容**
   - 确保使用 Python 3.10
   - 检查虚拟环境配置

2. **CUDA 版本问题**
   - 确认系统安装了 CUDA 12.1
   - 使用 `nvidia-smi` 检查 GPU 状态
   - 验证 PyTorch CUDA 支持：`python -c "import torch; print(torch.cuda.is_available())"`

### 常见问题

1. **CUDA 内存不足**
   - 减小 `per_device_train_batch_size`
   - 启用梯度累积
   - 使用 4bit 量化

2. **数据格式错误**
   - 检查原始数据文件完整性
   - 确认文件路径正确
   - 重新运行数据预处理

3. **API 调用失败**
   - 验证 API 密钥和 URL
   - 检查网络连接
   - 调整请求频率

4. **依赖安装问题**
   - 按照推荐顺序安装依赖
   - 使用虚拟环境避免版本冲突
   - 如果 `unsloth` 安装失败，尝试从源码安装

5. **vLLM 部署问题**
   - 确认模型路径正确
   - 检查模型文件完整性
   - 调整 `max_model_len` 参数适应显存大小
   - 如果启动失败，检查端口是否被占用
