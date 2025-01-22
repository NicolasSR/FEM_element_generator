import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from evaluate import load as eval_load
from safetensors.torch import load_file as load_safetensors_file

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

# test_latex = '\\nabla v \\cdot k \\nabla u'
# test_pseudocode = 'dot(grad(var0_0),prod(var0_1,grad(var0_2)))'

# test_latex = 'f v'
# test_pseudocode = 'prod(var0_0,var0_1)'

# test_latex = 'F U \\text{skew} \\left [\\left (\\left [\\boldsymbol{\\alpha}\\right ] \\otimes \\boldsymbol{\\alpha}\\right )^T:\\partial_{z}(\\boldsymbol{\\Psi})\\right ]'
# test_pseudocode = 'matrix_prod(matrix_prod(var2_0,var2_1),skew(contract(transpose(outer(var1_0,var1_0)),partial(var4_0,coord_2))))'

# test_latex = '\\nabla \\left (\\boldsymbol{\\Xi}:\\frac{\\boldsymbol{\\Phi}}{\\frac{\\partial (\\alpha)}{\\partial e_{j}}}\\right )'
# test_pseudocode = 'grad(contract(var4_0,frac(var4_1,partial(var0_0,coord_1))))'

test_latex = '\\nabla \\cdot \\left (\\nabla \\times (\\left (\\boldsymbol{\\beta}\\right )+\\boldsymbol{\\beta})\\right )'
test_pseudocode = 'div(curl(plus(var1_0,var1_0)))'

# test_latex = '(\\text{skew}\ (I)^T)^T'
# test_pseudocode = 'transpose(skew(transpose(var2_0)))'

# test_latex = 'e^{\\text{Tr}\ \\text{sym} [(\\boldsymbol{\\Pi}:\\boldsymbol{\\Phi}) A]}'
# test_pseudocode = 'exp(tr(sym(prod(contract(var4_0,var4_1),var2_0))))'

# test_latex = '\\text{sym}\ I \\left [\\boldsymbol{\\lambda} \\cdot \\mathbf{z}\\right ]'
# test_pseudocode = 'prod(sym(var2_0),dot(var1_0,var1_1))'

# test_latex = '\\mathbf{v}+\\mathbf{v}'
# test_pseudocode = 'plus(var1_0,var1_0)'

# test_latex = '\\exp{\\text{det}\ \\frac{\\partial \\left [\\frac{V}{\\boldsymbol{\\zeta} \\cdot (\\boldsymbol{\\zeta})}\\right ]}{\\partial e_{3}}}'
# test_pseudocode = 'exp(det(partial(frac(var2_0,dot(var1_0,var1_0)),coord_2)))'

# test_latex = '\\exp{\\text{sgn}\ \\sqrt{(\\boldsymbol{\\zeta}):(\\boldsymbol{\\psi}+\\boldsymbol{\\psi})}}'
# test_pseudocode = 'exp(sign(sqrt(contract(var1_0,plus(var1_1,var1_1)))))'

# test_latex = '\\text{dev}\ E'
# test_pseudocode = 'dev(var2_0)'


max_seq_length_encoder = 512
max_seq_length_decoder = 64

input_text = 'Generate the pseudocode for this mathematical expression written in LaTeX:\n' + test_latex
input_ids = bart_tokenizer(input_text, padding='max_length', max_length=max_seq_length_encoder, return_tensors="pt").input_ids
# print(input_ids)

# Sampled generation
# torch.manual_seed(42)
temp = 1.0
top_p = 0.9
do_sample = True
print('Temp: ', temp, '. Top_p: ', top_p, '. Do_sample: ', do_sample)

for i in range(10):
    outputs = custom_full_model.generate(input_ids, do_sample=do_sample, temperature=temp, top_p=top_p, max_new_tokens=max_seq_length_decoder, no_repeat_ngram_size=None)

    # Deterministic (greedy) generation
    # torch.manual_seed(42)
    # do_sample = False
    # num_beams=1
    # top_k=0
    # top_p=1.0
    # outputs = custom_full_model.generate(input_ids, do_sample=do_sample, num_beams=num_beams, top_k=top_k, top_p=top_p, max_new_tokens=max_seq_length_decoder, no_repeat_ngram_size=None)

    response = custom_tokenizer.batch_decode(outputs, skip_special_tokens=False)
    print(response)
