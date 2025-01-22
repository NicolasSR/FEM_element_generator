## Code taken from one of Unsloth's own examples at:
#  https://github.com/unslothai/unsloth?tab=readme-ov-file#-finetune-for-free

import numpy as np
import torch

from datasets import load_dataset, load_from_disk
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForSeq2Seq
from trl import DataCollatorForCompletionOnlyLM
from evaluate import load as eval_load

from unsloth import is_bfloat16_supported

max_seq_length = 512 # 512 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

model_name = 'meta-llama/Llama-3.2-1B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Configure token for our chat format
with open("LLM_trainer/custom_chat_template.jinja", "r") as jinja_file:
    custom_template = jinja_file.read()
tokenizer.chat_template = custom_template


print(f"Pad Token id: {tokenizer.pad_token_id} and Pad Token: {tokenizer.pad_token}")
print(f"EOS Token id: {tokenizer.eos_token_id} and EOS Token: {tokenizer.eos_token}")

tokenizer.pad_token='<|finetune_right_pad_id|>'
tokenizer.eos_token='<|eot_id|>'

# Example sample for our training set
# example_data_latex = '\\langle \\mathbf{U}+\\boldsymbol{\psi}:\\mathbf{B},\\mathbf{U} \\text{fun}(\\mathbf{U},\\mathbf{Z},\\mathbf{Z}) \\rangle'
# example_data_code = 'inner(plus(U_bold,contract(psi_bold,B_bold)),prod(U_bold,fun(U_bold,Z_bold,Z_bold)))'
# formatted_example = [{'role': 'user', 'content': example_data_latex},{'role': 'assistant', 'content': example_data_code}]

# print(tokenizer.apply_chat_template(formatted_example, tokenize = False, add_generation_prompt=False))
# print(len(tokenizer.apply_chat_template(formatted_example, tokenize = True, add_generation_prompt=False)))

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    # texts = [tokenizer.apply_chat_template(convo, tokenize = True, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
# def formatting_eval_data_func(examples):
#     convos = examples["conversations"]
#     texts = [tokenizer.apply_chat_template([convo[0]], tokenize = False, add_generation_prompt = True) for convo in convos]
#     # labels = [tokenizer.encode(convo[1]['content']) for convo in convos]
#     labels = [convo[1]['content'] for convo in convos]
#     return { "text" : texts, "label": labels}

# def formatting_prompts_func_tokens_count(examples):
#     tokens_lists = examples["text"]
#     tokens_counts = [len(tokens_list) for tokens_list in tokens_lists]
#     return { "tokens_count" : tokens_counts, }
# pass

dataset = load_from_disk("latex_pseudocode_dataset_10K.hf")
# dataset = load_from_disk("latex_pseudocode_dataset_noFun_10K.hf")
dataset_val = load_from_disk("latex_pseudocode_dataset_validation_1K.hf").select(range(16))
dataset = dataset.map(formatting_prompts_func, batched = True,)
dataset_val = dataset_val.map(formatting_prompts_func, batched = True,)
# dataset_val = dataset_val.map(formatting_eval_data_func, batched = True,)
# dataset = dataset.map(formatting_prompts_func_tokens_count, batched = True,)

# print(dataset_val['text'][0])
# print(dataset_val['label'][0])
# exit()

response_template = '<|start_header_id|>assistant<|end_header_id|>'
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# def compute_metrics(eval_preds):
#     logits, labels = eval_preds
#     # print(logits.shape)
#     # print(labels.shape)
#     # exit()
#     # metric = eval_load("glue", "mrpc")
#     predictions = np.argmax(logits, axis=-1)
#     # print(predictions)
#     # print(labels[0])
#     predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     # print(predictions)
#     # print(labels)
#     predictions_string = tokenizer.batch_decode(predictions, skip_special_tokens=False)
#     labels_string = tokenizer.batch_decode(labels, skip_special_tokens=False)
#     # print(predictions_string)
#     # print(labels_string)
#     # exit()
#     metric = eval_load("bleu")
#     return metric.compute(predictions=predictions_string, references=labels_string)
#     # return {'mymetric': 1.0}

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = dataset_val,
    # compute_metrics=compute_metrics,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = collator,
    # data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 1,
        per_device_eval_batch_size = 4,
        eval_accumulation_steps = 2,
        warmup_steps = 5,
        num_train_epochs = 50, # Set this for 1 full training run.
        # max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 12500,
        save_steps=31250,
        eval_strategy="steps",
        eval_steps = 12500,
        optim = "adamw_bnb_8bit", # "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407, # 3407
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

# trainer = train_on_responses_only(
#     trainer,
#     instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
#     response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
# )

#Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()
# trainer_stats = trainer.train(resume_from_checkpoint=True)