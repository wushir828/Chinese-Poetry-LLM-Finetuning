# Chinese-Poetry-LLM-Finetuning
使用LoRA对Qwen1.5-1.8B模型进行微调，以生成古典中文诗歌。本项目涵盖了从数据准备、微调训练到对比测试的完整工作流程。
# 使用LoRA微调大模型生成古典诗词 (Fine-tuning LLM with LoRA for Classical Chinese Poetry Generation)

这是一个端到端的项目，旨在展示如何使用PEFT（LoRA）技术，对一个预训练的通用大语言模型（Qwen1.5-1.8B-Chat）进行微调，使其能够学习并模仿中国古典诗词的风格进行创作。
---

## 🤖 微调后的模型 (Fine-tuned Model)

本项目微调后的LoRA适配器已上传至Hugging Face Hub，可以方便地加载用于推理或进一步的开发。

**模型链接**: [Your-HF-Username/Qwen1.5-1.8B-Poet-LoRA](https://huggingface.co/wushir828/Qwen1.5-1.8B-Chinese-Poet-LoRA)
## 最终成果展示<img width="1311" height="83" alt="Image" src="https://github.com/user-attachments/assets/f75e675c-0542-4f54-bfa7-22ff4273c7f0" /><img width="1430" height="92" alt="Image" src="https://github.com/user-attachments/assets/ac1e52f5-fd01-4615-abe0-471fe9189878" /><img width="1293" height="336" alt="Image" src="https://github.com/user-attachments/assets/3bbe09f7-2342-423d-ba67-347b196ae6f2" />
## TRAINNG LOSS
<img width="233" height="840" alt="Image" src="https://github.com/user-attachments/assets/52ba8b37-2c50-410a-84b6-635adba9fe1a" />
<img width="234" height="846" alt="Image" src="https://github.com/user-attachments/assets/9d80d908-3c16-4300-a1ba-d2386c83cbaa" />
---

## 📖 项目概述 (Project Overview)

本项目完整地实现了从数据准备、模型加载、配置微调参数到执行训练、以及最终通过A/B对比测试来验证微调效果的全过程。最终，我们得到了一个能够创作出具有古典韵味文本的、专属的“诗人模型”。

本项目完整地实现了以下工作流程：
1.  **数据准备**: 从Kaggle加载并处理一份中国古典诗词数据集。
2.  **模型微调**: 配置LoRA参数，并使用 transformers 和 trl 库对基础模型进行微调。 
3.  **效果评估**: 通过对原始基础模型和微调后模型进行A/B对比测试，来验证训练的有效性。

---
## 🛠️ 技术栈 (Tech Stack)

* **核心框架**: PyTorch
* **模型与训练**: Hugging Face `transformers`, `datasets`, `peft`, `trl`
* **运行环境**: Google Colab (T4 GPU)
* **基础模型**: `Qwen/Qwen1.5-1.8B-Chat`
---
## 🚀 工作流程 (Workflow)

本项目的所有代码均在 `[你的ipynb文件名].ipynb` 文件中。

### 第1步：获取数据集 (Get the Dataset)

本项目使用了Kaggle上的一个公开数据集，包含了5000首中国古典诗词。

* **数据集名称**: Enhanced Classical Chinese Poetry Dataset
* **Kaggle链接**: [https://www.kaggle.com/datasets/zyan1999/enhanced-classical-chinese-poetry-dataset](https://www.kaggle.com/datasets/zyan1999/enhanced-classical-chinese-poetry-dataset)

**操作步骤**:
1.  从上述链接下载数据集 (`archive.zip`)。
2.  解压文件，得到 `Chinese_Poetry.csv`。
3.  在Google Colab环境中，将此`.csv`文件上传，代码将读取该文件进行处理。

### 第2步：模型微调 (Model Fine-tuning)

微调过程利用了LoRA技术，以在消费级GPU上实现高效的参数训练。

1.  **加载基础模型**: 我选择了清华大学开源的 `Qwen/Qwen1.5-1.8B-Chat` 作为基础模型。代码中使用了`bitsandbytes`库进行4-bit量化加载，以大幅降低显存占用。
2.  **配置LoRA**: 使用`peft`库来配置LoRA参数，仅在Transformer的注意力层（`q_proj`, `k_proj`, `v_proj`, `o_proj`）上应用可训练的“适配器”。
3.  **执行训练**: 使用`trl`库中的`SFTTrainer`，这是一个为监督式微调优化的训练器，它简化了训练循环的编写。训练过程在Google Colab的免费T4 GPU上进行。

### 第3步：A/B对比测试 (A/B Comparison)

为了直观地验证微调的效果，代码的最后一部分包含了一个A/B对比测试。

1.  **加载两个模型**:
    * **A组 (实验组)**: 我们刚刚微调好的模型。
    * **B组 (对照组)**: 重新加载一个未经任何修改的、原始的 `Qwen/Qwen1.5-1.8B-Chat` 模型。
2.  **进行测试**:
    * 使用完全相同的提示（Prompt），例如“床前明月光，”，分别输入给两个模型。
    * 比较两个模型的输出，以评估微调后的模型是否在风格上更接近古典诗词。

---
### 调优过程与思考 (Tuning Process & Reflections)

在项目的初次探索阶段，我将训练步数 `max_steps` 设置为 **100** 步，希望能快速验证整个微调流程的可行性。

训练完成后，通过A/B对比测试发现，虽然微调后的模型（实验组）在回答上与原始模型（对照组）产生了一定的差异，但其生成的文本风格依然非常接近现代白话文，未能很好地学习到古典诗词的韵律和风格。

**[100步训练时模型的测试结果截图]**<img width="1662" height="573" alt="Image" src="https://github.com/user-attachments/assets/7cfc3135-32e2-4986-9458-2e446d095de6" />

为了解决这个问题，我分析其根本原因是**训练力度不足**。对于一个拥有数十亿参数的基础模型来说，仅仅100步的学习并不足以克服其原有的“对话惯性”，也无法让它充分学习到新风格的精髓。

因此，我进行了第二轮实验，将 `max_steps` 的值从 `100` 增加到了 **`500`**，进行了更充分的训练。

从最上方的成果展示以及新的A/B对比测试结果可以看出，经过500步的微调，模型的输出在用词、句式和整体意境上，都表现出了远胜于初版模型的古典诗词风格，证明了**增加训练步数对于风格迁移任务的有效性**。
这个迭代过程表明，模型微调不仅是代码的执行，更是一个需要不断评估结果、分析问题、并通过调整超参数来持续优化的科学实验过程。
