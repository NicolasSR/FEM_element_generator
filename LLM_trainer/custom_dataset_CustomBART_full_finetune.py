import numpy as np

import torch
from safetensors.torch import load_file as load_safetensors_file
from datasets import load_from_disk

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

# from custom_decoder_model.custom_model_configuration_explicit_vars import CustomConfig
from custom_decoder_model.custom_model_configuration_symbolic_vars import CustomConfig
from custom_decoder_model.custom_tokenizer_test import CustomTokenizer
from custom_decoder_model.custom_full_model_test import CustomBartForConditionalGeneration
from custom_decoder_model.custom_data_collators import DataCollatorForDualTokenizerSeq2Seq, DataCollatorForDualTokenizerSeq2SeqCausalCollapse


bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
# custom_tokenizer = CustomTokenizer(vocab_file="LLM_trainer/custom_decoder_model/vocab_explicit_vars.json")
custom_tokenizer = CustomTokenizer(vocab_file="LLM_trainer/custom_decoder_model/vocab_symbolic_vars.json")

# bart_full_model = AutoModelForSeq2SeqLM.from_pretrained("BART_TestFineTunedCheckpoint")
# print(id(bart_full_model.model.encoder.embed_tokens.weight))

encoder_config = AutoConfig.from_pretrained(pretrained_model_name_or_path='LLM_trainer/custom_decoder_model/encoder_config.json')
# decoder_config = CustomConfig.from_pretrained(pretrained_model_name_or_path='LLM_trainer/custom_decoder_model/decoder_config_explicit_vars_384.json')
decoder_config = CustomConfig.from_pretrained(pretrained_model_name_or_path='LLM_trainer/custom_decoder_model/decoder_config_symbolic_vars_768.json')
custom_full_model = CustomBartForConditionalGeneration(encoder_config, decoder_config, dual_embedding_size=False)

""" 
## PHASE 1 (only encoder weights available from finetuned. This will train only the decoder)

safetensors_weights = load_safetensors_file("BART_TestFineTunedCheckpoint/model.safetensors")

with torch.no_grad():
    for key, tensor in safetensors_weights.items():

        if key == 'model.shared.weight':
            model_key = 'embed_tokens.weight'
            print(f"Loading weights from {key} in safetensors file for: custom_full_model.model.encoder.{model_key}")
            custom_full_model.model.encoder.state_dict()[model_key].copy_(tensor)
            continue

        # Map the SafeTensors keys to model keys
        model_key = key.replace("model.encoder.", "")
        # model_key = key.replace("model.encoder.", "custom_full_model.model.encoder.")
        
        # Check if the key exists in the model's state_dict
        if model_key in custom_full_model.model.encoder.state_dict():
            print(f"Loading weights for: custom_full_model.model.encoder.{model_key}")
            custom_full_model.model.encoder.state_dict()[model_key].copy_(tensor)
        else:
            # print(f"Key {model_key} not found in encoder model's state_dict!")
            pass

print(id(custom_full_model.model.encoder.embed_tokens.weight))
print(id(custom_full_model.model.decoder.embed_tokens.weight))
print(id(custom_full_model.lm_head.weight))

for param in custom_full_model.model.encoder.parameters():
    param.requires_grad = False
# # Check frozen parameters
# for name, param in custom_full_model.named_parameters():
#     print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

##
"""


## PHASE 2 (All custom model weights available from finetuned model)

safetensors_weights = load_safetensors_file("CUSTOM_FineTunedCheckpoint_Phase1/model.safetensors")

with torch.no_grad():
    for key, tensor in safetensors_weights.items():
        # Map the SafeTensors keys to model keys
        
        # Check if the key exists in the model's state_dict
        if key in custom_full_model.state_dict():
            print(f"Loading weights for: {key}. ",tensor.shape)
            custom_full_model.state_dict()[key].copy_(tensor)
        else:
            print(f"Key {key} not found in encoder model's state_dict!")
            pass

##



max_seq_length_encoder = 512
# max_seq_length_decoder = 128
max_seq_length_decoder = 64

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    input_texts = ['Generate the pseudocode for this mathematical expression written in LaTeX:\n' + convo[0]['content'] for convo in convos]
    label_texts = [convo[1]['content'] for convo in convos]

    model_inputs = bart_tokenizer(input_texts, max_length=max_seq_length_encoder, truncation=True)
    model_labels = custom_tokenizer(label_texts, max_length=max_seq_length_decoder, truncation=True)

    model_inputs["labels"] = model_labels["input_ids"]
    # model_inputs["id_labels"] = model_labels["input_ids"]

    return model_inputs

dataset = load_from_disk("latex_pseudocode_dataset_train_10K.hf")
dataset_val = load_from_disk("latex_pseudocode_dataset_validation_1K.hf")#.select(range(16))
dataset = dataset.map(formatting_prompts_func, batched = True,)
dataset_val = dataset_val.map(formatting_prompts_func, batched = True,)

training_args = Seq2SeqTrainingArguments(
    output_dir="custom_model_output",
    learning_rate=2e-4,
    lr_scheduler_type = "linear",
    warmup_steps = 5,
    num_train_epochs=500,
    per_device_train_batch_size=8,
    gradient_accumulation_steps = 1,
    per_device_eval_batch_size=8,
    eval_accumulation_steps = 2,
    weight_decay=0.01,
    optim = "adamw_bnb_8bit", # "adamw_8bit",
    eval_strategy="steps",
    eval_steps = 12500,
    logging_steps = 12500,
    save_steps=125000,
    save_total_limit=2,
    # load_best_model_at_end=True,
    predict_with_generate=True,
    bf16=True,
    fp16=False,
    seed = 3407,
)

# data_collator = DataCollatorForSeq2Seq(model=custom_full_model, tokenizer=bart_tokenizer, padding='max_length', max_length=max_seq_length_encoder)
# data_collator = DataCollatorForDualTokenizerSeq2Seq(model=custom_full_model,
data_collator = DataCollatorForDualTokenizerSeq2SeqCausalCollapse(model=custom_full_model,
                                                    encoder_tokenizer=bart_tokenizer,
                                                    decoder_tokenizer=custom_tokenizer,
                                                    padding='max_length',
                                                    encoder_max_length=max_seq_length_encoder,
                                                    decoder_max_length=max_seq_length_decoder)
# data_collator(dataset_val)

""" # Convert to list of dicts with tensors
batch = [{"input_ids": item["input_ids"], "labels": item["labels"]} for item in dataset.select(range(10))]
# Generate a batch
batch_output = data_collator(batch)

for i in range(len(batch)):
    print()
    print(batch[i]['labels'])
    print(batch_output['decoder_input_ids'][i])
    print(batch_output['labels'][i])
    # print(batch_output.keys())
exit() """

trainer = Seq2SeqTrainer(
    model = custom_full_model,
    tokenizer = None,
    train_dataset = dataset,
    eval_dataset = dataset_val,
    # compute_metrics=compute_metrics,
    data_collator = data_collator,
    args = training_args
)

trainer_stats = trainer.train()

# test_latex = '\\langle \\boldsymbol{\\tau} \\otimes \\mathbf{V},\\boldsymbol{\\Upsilon} \\rangle \\cdot \\sgn{\\langle \\mathbf{C},\\mathbf{V} \\rangle}'
# test_pseudocode = 'prod(inner(outer(tau_bold,V_bold),Upsilon_bold),sign(inner(C_bold,V_bold)))'

# input_ids = bart_tokenizer.encode(test_latex, return_tensors='pt')
# print(input_ids)

# output_ids = custom_full_model.generate(input_ids=input_ids, tokenizer=custom_tokenizer)
# output_ids = custom_tokenizer.batch_decode(output_ids)
# print(output_ids)


