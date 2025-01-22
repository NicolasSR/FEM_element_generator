import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from evaluate import load as eval_load
from safetensors.torch import load_file as load_safetensors_file
from datasets import load_from_disk

from custom_decoder_model.custom_model_configuration_symbolic_vars import CustomConfig
from custom_decoder_model.custom_tokenizer_test import CustomTokenizer
from custom_decoder_model.custom_full_model_test import CustomBartForConditionalGeneration


bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
custom_tokenizer = CustomTokenizer(vocab_file="LLM_trainer/custom_decoder_model/vocab_symbolic_vars.json")

encoder_config = AutoConfig.from_pretrained(pretrained_model_name_or_path='LLM_trainer/custom_decoder_model/encoder_config.json')
decoder_config = CustomConfig.from_pretrained(pretrained_model_name_or_path='LLM_trainer/custom_decoder_model/decoder_config_symbolic_vars_768.json')
custom_full_model = CustomBartForConditionalGeneration(encoder_config, decoder_config, dual_embedding_size=False)

safetensors_weights = load_safetensors_file("CUSTOM_FineTunedCheckpoint_Phase2/model.safetensors")

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

total_params = sum(p.numel() for p in custom_full_model.parameters())
print(total_params)
print(id(custom_full_model.model.encoder.embed_tokens.weight))
print(id(custom_full_model.model.decoder.embed_tokens.weight))
print(id(custom_full_model.lm_head.weight))


# test_latex = '\\delta \\cdot \\boldsymbol{\\Theta} \\cdot \\partial_{e_{3}}(\\text{fun} (\\mathbf{S}+(\\mathbf{K}),\\mathbf{z},\\mathbf{S}))'
# test_pseudocode = 'inner(prod(var0_0,var4_0),partial(fun(plus(var3_0,var3_1),var1_0,var3_0),coord_2))'


max_seq_length_encoder = 512
max_seq_length_decoder = 64


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    input_texts = ['Generate the pseudocode for this mathematical expression written in LaTeX:\n' + convo[0]['content'] for convo in convos]
    label_texts = [convo[1]['content'] for convo in convos]

    model_inputs = bart_tokenizer(input_texts, max_length=max_seq_length_encoder, truncation=True)
    model_labels = custom_tokenizer(label_texts, max_length=max_seq_length_decoder, truncation=True)

    model_inputs["labels"] = model_labels["input_ids"]

    return model_inputs

dataset_test = load_from_disk("latex_pseudocode_dataset_test_1K.hf") # .select(range(10))
dataset_test = dataset_test.map(formatting_prompts_func, batched = True,)

# Sampled generation
# torch.manual_seed(42)
temp = 1.0
top_p = 0.9
do_sample = True
print('Temp: ', temp, '. Top_p: ', top_p, '. Do_sample: ', do_sample)

generated_outputs_list = []
for i in range(dataset_test.num_rows):
    if i%100 == 0:
        print(i)
    generation_input = torch.tensor(dataset_test['input_ids'][i], dtype=torch.int64)
    generation_input = generation_input.repeat(10, 1)
    outputs = custom_full_model.generate(generation_input, do_sample=do_sample, temperature=temp, top_p=top_p, max_new_tokens=max_seq_length_decoder, no_repeat_ngram_size=None)

    # print(outputs)

    generated_outputs_list.append(outputs.detach().tolist())

print(len(generated_outputs_list))
dataset_test = dataset_test.add_column('generation_output_ids', generated_outputs_list)
print(dataset_test)

dataset_test.save_to_disk("test_dataset_with_outputs.hf")
