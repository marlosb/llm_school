# LLM School 🎓

Welcome to **LLM School** - a comprehensive learning project for exploring, understanding, and experimenting with Large Language Models (LLMs). This repository serves as a hands-on educational journey through the internals, creation, fine-tuning, and practical applications of modern language models.

## 🎯 Project Objectives

This project is designed to provide deep, practical understanding of:

- **LLM Architecture & Internals**: Understanding transformer architecture, attention mechanisms, embeddings, and layer-by-layer operations
- **Model Exploration**: Hands-on experimentation with pre-trained models from Hugging Face
- **Fine-tuning Techniques**: Learning various approaches to adapt models for specific tasks
- **Model Creation**: Building custom models from scratch
- **Performance Optimization**: Techniques for efficient training and inference

## 📚 Learning Path

### 101 - Explore LLM
**Current Status**: 🚧 In Progress

- **[01 DISTILGPT2](./101%20Explore%20LLM/01%20DISTILGPT2.ipynb)**: Deep dive into DistilGPT2 architecture
  - Tokenization and encoding/decoding processes
  - Word Token Embeddings (wte) - understanding vocabulary and embedding vectors
  - Word Position Embeddings (wpe) - positional information encoding
  - Dropout layers and regularization
  - Transformer blocks and layer normalization
  - Self-attention mechanisms
  - *Status*: Exploring attention layer components (Q, K, V vectors)

### Planned Modules

- **102 - Fine-tuning Techniques**
  - Transfer learning concepts
  - Parameter-efficient fine-tuning (LoRA, QLoRA)
  - Task-specific adaptations

  - **103 - Model Training Fundamentals**
  - Training loops and optimization
  - Loss functions for language modeling
  - Gradient computation and backpropagation

- **104 - Advanced Architectures**
  - Comparing different transformer variants
  - Attention mechanisms deep dive
  - Model scaling and efficiency

- **105 - Custom Model Creation**
  - Building models from scratch
  - Custom tokenizers
  - Training pipeline implementation

## 🛠️ Setup & Requirements

### Prerequisites
- Python 3.8+
- PyTorch
- Transformers library
- Jupyter Notebook/Lab
- Hugging Face account (for model access)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/marlosb/llm_school.git
cd llm_school
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Hugging Face token:
   - Create an environment variable with Hugging Face token: `HUGGINGFACE_HUB_TOKEN=your_token_here`


## 🔍 Current Focus: DistilGPT2 Exploration

The project currently explores **DistilGPT2**, a distilled version of OpenAI's GPT-2:
- **Original GPT-2**: 12 transformer blocks, 124M parameters
- **DistilGPT2**: 6 transformer blocks, 82M parameters
- **Benefit**: Faster inference while maintaining good performance

### Key Learning Areas (In Progress)
1. **Tokenization**: Understanding how text is converted to tokens and back
2. **Embeddings**: Word token embeddings (50,257 vocabulary) and positional embeddings (1,024 max sequence length)
3. **Transformer Architecture**: Layer-by-layer analysis of the 6-block structure
4. **Attention Mechanism**: Deep dive into self-attention, Q/K/V vectors, and attention matrices

## 🎓 Learning Resources

### Recommended Reading
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108) - Knowledge distillation techniques

### Useful Tools
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) - Visualize tokenization
- [Hugging Face Model Hub](https://huggingface.co/models) - Pre-trained models
- [Papers With Code](https://paperswithcode.com/methods/category/transformers) - Latest research

## 🤝 Contributing

This is a personal learning project, but contributions are welcome! Feel free to:
- Suggest improvements to existing notebooks
- Propose new learning modules
- Share interesting findings or experiments
- Report issues or unclear explanations

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Connect

- **GitHub**: [marlosb](https://github.com/marlosb)
- **Project Repository**: [llm_school](https://github.com/marlosb/llm_school)

---

> **Note**: This project is for educational purposes. Always respect model licenses and terms of use when working with pre-trained models.
