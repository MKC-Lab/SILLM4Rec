# SILLM4Rec: Sequential Item Recommendation with Large Language Models

## 🚀 Quick Start

### 1. Environment Setup

#### 1.1 System Requirements

- **Python**: 3.10
- **CUDA**: 12.1
- **GPU**: Recommended to use GPU with large memory (e.g., A100 40G)

#### 1.2 Installing Dependencies

**Important**: Please install dependencies in the following order to ensure compatibility.

First install PyTorch GPU version (CUDA 12.1):

```bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Then install other dependencies:

```bash
pip install -r requirements.txt
```

Verify CUDA availability:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
```

### 2. Data Preparation

#### 2.1 Download Raw Data

Visit the [Amazon Review Data 2023](https://amazon-reviews-2023.github.io/) website to download the following files (5 core):

- **Metadata files**: `meta_[CATEGORY].jsonl` (e.g., `meta_Baby_Products.jsonl`)
- **Review data files**: `[CATEGORY].jsonl` (e.g., `Baby_Products.jsonl`)  
- **Interaction data files**: `[CATEGORY].test.csv` (e.g., `Baby_Products.test.csv`)

Place the downloaded files in the `raw_data/` directory.

#### 2.2 Data Preprocessing

Use `raw_data_process.ipynb` to process the raw data:

1. Open `raw_data_process.ipynb`
2. Modify file paths according to your data category
3. Run all cells to generate processed data

Preprocessing steps include:

- Generate user and item mapping relationships
- Convert JSONL format to Parquet format
- Filter high-quality user interaction data
- Generate training and test sets

### 3. Training Data Generation

Use `training_data_process.ipynb` to generate training data and evaluation results:

1. Configure API keys and base URL (for calling large language models)
2. Run image description generation (convert product images to text descriptions)
3. Generate user preference summaries
4. Create candidate product ranking tasks
5. Generate SFT and DPO training data

Main features:

- **Image-to-text conversion**: Use vision-language models to describe product images
- **User preference analysis**: Generate user preference summaries based on historical interactions
- **Ranking task generation**: Create product ranking tasks and corresponding standard answers
- **Training data construction**: Generate data formats suitable for different training methods

### 4. Model Preparation

#### 4.1 Download Base Model

Before starting training, you need to download and prepare the base model:

```bash
# Download model to local directory, for example:
# models/DeepSeek-R1-Distill-Qwen-7B/
# Or download using Hugging Face Hub
```

Ensure model files are complete and placed in the appropriate directory.

### 5. Model Training

#### 5.1 Supervised Fine-Tuning (SFT)

Use `SFT.ipynb` for supervised fine-tuning:

1. Configure base model path in the notebook
2. Load pretrained model (e.g., DeepSeek-R1-Distill-Qwen-7B)
3. Configure LoRA parameters
4. Load SFT training data
5. Start training

Features:

- Support 4-bit quantization loading
- Use LoRA for efficient fine-tuning
- Automatic gradient checkpointing
- Support multiple optimizers

#### 5.2 Direct Preference Optimization (DPO)

Use `DPO.ipynb` for preference optimization:

1. Load the SFT fine-tuned model
2. Configure DPO training parameters
3. Load DPO training data (including chosen and rejected samples)
4. Start DPO training

Features:

- Further optimization based on SFT model
- Automatic handling of preference pair data
- Support custom β parameter
- Memory efficiency optimization

### 6. Model Deployment

#### 6.1 Install vLLM

After training is complete, it's recommended to use vLLM for efficient inference deployment:

```bash
pip install vllm
```

#### 6.2 Start Inference Service

Use the following command to start the vLLM inference service:

```bash
vllm serve {model_location} --max_model_len=4096 --override-generation-config "{\"temperature\": 0.2}"
```

Examples:

```bash
# Deploy SFT model
vllm serve ./new_model/qwen-sft --max_model_len=4096 --override-generation-config "{\"temperature\": 0.2}"

# Deploy DPO model
vllm serve ./new_model/qwen-dpo --max_model_len=4096 --override-generation-config "{\"temperature\": 0.2}"
```

#### 6.3 Using Inference Service

After the vLLM service is started, you can make inference calls through the HTTP API:

```python
api_key = "EMPTY"
base_url = "http://localhost:8000/v1"
```

## 📁 Project Structure

```text
SILLM4Rec/
├── raw_data/                    # Raw data directory
│   ├── meta_[CATEGORY].jsonl    # Product metadata
│   ├── [CATEGORY].jsonl         # User review data
│   └── [CATEGORY].test.csv      # Interaction test data
├── [category]_data/             # Processed data directory
│   ├── train/                   # Training data
│   └── test/                    # Test data
├── raw_data_process.ipynb       # Data preprocessing
├── training_data_process.ipynb  # Training data generation
├── SFT.ipynb                    # Supervised fine-tuning
├── DPO.ipynb                    # Direct preference optimization
├── matrix.py                    # Evaluation metrics calculation
├── requirements.txt             # Dependencies list
└── README.md                    # Project documentation
```

## 🔧 Configuration

### API Configuration

Configure your LLM API in `training_data_process.ipynb`:

```python
api_key = "your_api_key_here"
base_url = "your_base_url_here"
```

### Model Path Configuration

Modify paths in each notebook according to your model storage location:

```python
# In SFT.ipynb - Base model path
model_name = "/path/to/your/base/model"  # Example: "./models/DeepSeek-R1-Distill-Qwen-7B"

# In DPO.ipynb - SFT fine-tuned model path  
model_name = "/path/to/your/sft/model"   # Example: "./new_model/qwen-sft"

# Training output path
output_dir = "/path/to/output"           # Example: "./outputs"
save_dir = "/path/to/save/model"        # Example: "./new_model/qwen-sft"
```

### Recommended Directory Structure

```text
SILLM4Rec/
├── models/                      # Base models directory
│   └── DeepSeek-R1-Distill-Qwen-7B/
├── new_model/                   # Fine-tuned models directory
│   ├── qwen-sft/               # SFT model
│   └── qwen-dpo/               # DPO model
├── outputs/                     # Training output directory
└── ...                         # Other project files
```

## 📊 Evaluation Metrics

The project supports the following evaluation metrics:

- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Valid Rate**: Valid recommendation rate

## 🛠️ Troubleshooting

### Environment Issues

1. **Python Version Incompatibility**
   - Ensure you're using Python 3.10
   - Check virtual environment configuration

2. **CUDA Version Issues**
   - Confirm system has CUDA 12.1 installed
   - Use `nvidia-smi` to check GPU status
   - Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `per_device_train_batch_size`
   - Enable gradient accumulation
   - Use 4-bit quantization

2. **Data Format Errors**
   - Check raw data file integrity
   - Confirm file paths are correct
   - Re-run data preprocessing

3. **API Call Failures**
   - Verify API key and URL
   - Check network connection
   - Adjust request frequency

4. **Dependency Installation Issues**
   - Install dependencies in recommended order
   - Use virtual environment to avoid version conflicts
   - If `unsloth` installation fails, try installing from source

5. **vLLM Deployment Issues**
   - Confirm model path is correct
   - Check model file integrity
   - Adjust `max_model_len` parameter to fit GPU memory
   - If startup fails, check if port is already in use
