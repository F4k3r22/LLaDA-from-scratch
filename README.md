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

## 🛠️ Implementation Notes

It's important to clarify that the base architecture of the model (defined in [`modeling_llada.py`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/blob/main/modeling_llada.py)) already contains all necessary components to initialize and run LLaDA. Therefore, there is no need to redesign or significantly modify the model architecture itself.

The main objective of this project is to:

* Reuse the existing architecture.
* Implement the **data preparation pipeline** (tokenization, masking, and formatting).
* Create the **training loop** from scratch.
* Optionally, implement a **fine-tuning script** for adapting the pretrained model to specific downstream tasks.

## 💾 Dataset
The dataset we will use for training will be [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) in the `sample-10BT` subset

When the dataset is tokenized and masked, the dataset link will be provided so you can use it.

## 🙏 Acknowledgments

- The [ML-GSAI](https://github.com/ML-GSAI) team for the original architecture