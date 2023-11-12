# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # (C)Transformers trials

# %% [markdown]
# ## Setup

# %%
import huggingface_hub
import torch
import transformers
from transformers import AutoTokenizer

# %%
MODEL_PATHS = {
    "llama-2-13b-chat.ggmlv3.q8_0": "C:/Users/mehdi/Downloads/Models/llama-2-13b-chat.ggmlv3.q8_0.bin",
    "llama-2-7b-chat": "C:/Users/mehdi/Downloads/Models/llama/llama-2-7b-chat",
}

# %% [markdown]
# ## Text Generation

# %%
# from ctransformers import AutoModelForCausalLM

# llm = AutoModelForCausalLM.from_pretrained(
#     model_path_or_repo_id=MODEL_PATHS["llama-2-13b-chat.ggmlv3.q8_0"],
#     model_type='llama'
# )

# print(llm('AI is going to'))

# %% [markdown]
# have a significant impact on the future of work and will disrupt many industries. Here are some potential impacts of AI on various sectors:
# 1. Healthcare: AI has already made inroads into healthcare, with applications such as medical imaging analysis and drug discovery. As AI becomes more advanced, it could lead to personalized medicine, where treatments are tailored to individual patients based on their genetic profiles and medical history.
# 2. Finance: AI is being used in finance for fraud detection, credit scoring, and portfolio management. It could potentially automate many back-office functions, freeing up human professionals to focus on higher-level tasks.
# 3. Education: AI could revolutionize education by providing personalized learning experiences tailored to individual students' needs and abilities. It could also help teachers by automating administrative tasks and grading.
# 4. Manufacturing: AI is already being used in manufacturing to optimize production processes, predict maintenance needs, and improve product quality. As it becomes more advanced, it could lead to the development of smart factories where machines and robots work together seamlessly.
# 5. Transportation: AI is

# %%
huggingface_hub.notebook_login()

# %%
# tokenizer = LlamaTokenizer.from_pretrained("C:/Users/mehdi/Downloads/Models/llama/tokenizer")
# model = LlamaForCausalLM.from_pretrained(MODEL_PATHS["llama-2-7b-chat"])

# %%
# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf")

# %%
model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "meta-llama/Llama-2-13b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
