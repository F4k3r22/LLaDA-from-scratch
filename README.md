# 🚀 LLaDA-from-scratch

💡 I'm curious to implement LLaDA from scratch (For now I'm going to use what's in: [huggingface-modeling](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/blob/main/modeling_llada.py) as a base, from there I'll implement a 1B model to be trained from scratch)

LLaDA is a diffusion model for natural language that, unlike traditional autoregressive models, learns to model the distribution of text through a process of progressive data masking and its inverse reconstruction.

## 🔤 Tokenizer

To reduce complexity and implement LLaDA more quickly from scratch, we are going to reuse the tokenizer from [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)

## 📚 References for building LLaDA 1B from scratch

- [LLaDA-GitHub](https://github.com/ML-GSAI/LLaDA)
- [LLaDA-v1.5-GitHub](https://github.com/ML-GSAI/LLaDA-1.5)
- [LLaDA-Paper](https://arxiv.org/abs/2502.09992)
- [LLaDA-v1.5-Paper](https://arxiv.org/abs/2505.19223)
- [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/tree/main)
- [GSAI-ML/LLaDA-1.5](https://huggingface.co/GSAI-ML/LLaDA-1.5/tree/main)

## 🙏 Acknowledgments

- The [ML-GSAI](https://github.com/ML-GSAI) team for the original architecture