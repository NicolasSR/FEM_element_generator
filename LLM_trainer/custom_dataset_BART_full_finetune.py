from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

# assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."

# print(model.config)

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

# outputs = model.generate(inputs['input_ids'], max_new_tokens=512)
# outputs_string = tokenizer.batch_decode(outputs, skip_special_tokens=False)
# print(outputs_string)

max_seq_length = 512

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    input_texts = ['Generate the pseudocode for this mathematical expression written in LaTeX:\n' + convo[0]['content'] for convo in convos]
    label_texts = [convo[1]['content'] for convo in convos]

    model_inputs = tokenizer(input_texts, max_length=max_seq_length, truncation=True)
    model_labels = tokenizer(label_texts, max_length=max_seq_length, truncation=True)

    model_inputs["labels"] = model_labels["input_ids"]

    return model_inputs
    # return { "input_text" : input_texts, "output_text": label_texts, "label": model_labels}

dataset = load_from_disk("latex_pseudocode_dataset_10K.hf")
dataset_val = load_from_disk("latex_pseudocode_dataset_validation_1K.hf").select(range(16))
dataset = dataset.map(formatting_prompts_func, batched = True,)
dataset_val = dataset_val.map(formatting_prompts_func, batched = True,)

training_args = Seq2SeqTrainingArguments(
    output_dir="bart_output",
    learning_rate=2e-4,
    lr_scheduler_type = "linear",
    warmup_steps = 5,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps = 1,
    per_device_eval_batch_size=4,
    eval_accumulation_steps = 2,
    weight_decay=0.01,
    optim = "adamw_bnb_8bit", # "adamw_8bit",
    eval_strategy="steps",
    eval_steps = 12500,
    logging_steps = 12500,
    save_steps=12500,
    save_total_limit=2,
    load_best_model_at_end=True,
    predict_with_generate=True,
    bf16=True,
    fp16=False,
    seed = 3407,
)

data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer, padding='max_length', max_length=max_seq_length)

trainer = Seq2SeqTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = dataset_val,
    # compute_metrics=compute_metrics,
    data_collator = data_collator,
    args = training_args
)

trainer_stats = trainer.train()