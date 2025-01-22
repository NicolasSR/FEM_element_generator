import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from evaluate import load as eval_load
from safetensors.torch import load_file as load_safetensors_file

from custom_decoder_model.custom_model_configuration_explicit_vars import CustomConfig
from custom_decoder_model.custom_tokenizer_test import CustomTokenizer
from custom_decoder_model.custom_full_model_test import CustomBartForConditionalGeneration


bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
# custom_tokenizer = CustomTokenizer(vocab_file="LLM_trainer/custom_decoder_model/vocab_explicit_vars.json")
custom_tokenizer = CustomTokenizer(vocab_file="LLM_trainer/custom_decoder_model/vocab_symbolic_vars.json")

encoder_config = AutoConfig.from_pretrained(pretrained_model_name_or_path='LLM_trainer/custom_decoder_model/encoder_config.json')
decoder_config = CustomConfig.from_pretrained(pretrained_model_name_or_path='LLM_trainer/custom_decoder_model/decoder_config_explicit_vars_768.json')
# decoder_config = CustomConfig.from_pretrained(pretrained_model_name_or_path='LLM_trainer/custom_decoder_model/decoder_config_explicit_vars_384.json')
custom_full_model = CustomBartForConditionalGeneration(encoder_config, decoder_config, dual_embedding_size=False)

safetensors_weights = load_safetensors_file("CUSTOM_TestFineTunedCheckpoint_Phase2_EXPLICIT_VARS/model.safetensors")
# safetensors_weights = load_safetensors_file("custom_model_output/checkpoint-25000/model.safetensors")
# safetensors_weights = load_safetensors_file("CUSTOM384_TestFineTunedCheckpoint_Phase1/model.safetensors")

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

test_latex = '\\nabla v \\cdot k \\nabla u'
test_pseudocode = 'dot(grad(v_bold),prod(k,grad(u_bold)))'

# test_latex = 'f v'
# test_pseudocode = ''

# test_latex = '\\langle \\boldsymbol{\\tau} \\otimes \\mathbf{V},\\boldsymbol{\\Upsilon} \\rangle \\cdot \\sgn{\\langle \\mathbf{C},\\mathbf{V} \\rangle}'
# test_pseudocode = 'prod(inner(outer(tau_bold,V_bold),Upsilon_bold),sign(inner(C_bold,V_bold)))'

# test_latex = '\\text{fun}(\\mathbf{q},\\boldsymbol{\\Xi},\\mathbf{O},\\boldsymbol{\\Upsilon}) \\cdot \\mathbf{q} \\boldsymbol{\\psi} \\cdot \\varepsilon \\mathbf{q}'
# test_pseudocode = 'prod(inner(fun(q_bold,Xi_bold,O_bold,Upsilon_bold),q_bold),dot(psi_bold,prod(varepsilon,q_bold)))'

# test_latex = '\\sign{\\sigma} \\cdot \\sqrt{\\sign{\\sigma}}'
# test_pseudocode = 'prod(sign(sigma),sqrt(sign(sigma)))'

# test_latex = '\\sym{\\boldsymbol{\\Gamma}} \\cdot \\text{fun}(\\boldsymbol{\\Gamma},\\boldsymbol{\\Lambda} \\otimes \\mathbf{f},\\mathbf{I},\\boldsymbol{\\lambda})'
# test_pseudocode = 'inner(sym(Gamma_bold),fun(Gamma_bold,outer(Lambda_bold,f_bold),I_bold,lambda_bold))'

# test_latex = '\\sign{(\\boldsymbol{\\Delta}:\\mathbf{N})} \\cdot x'
# test_pseudocode = 'prod(sign(contract(Delta_bold,N_bold)),x)'

# test_latex = '\\nabla d \\cdot (\\mathbf{h}+\\frac{\\boldsymbol{\\rho}}{\\pi}) j'
# test_pseudocode = 'dot(grad(d),prod(plus(h_bold,frac(rho_bold,pi)),j))'

# test_latex = '\\langle \\frac{\\boldsymbol{\\Theta}}{n},\\text{fun}(\\mathbf{O},n,\\mathbf{j},\\mathbf{P}) \\rangle'
# test_pseudocode = 'inner(frac(Theta_bold,n),fun(O_bold,n,j_bold,P_bold))'

# test_latex = '\\frac{\\mathbf{y}+\\mathbf{y}}{\\nabla \\cdot \\mathbf{d}} \\cdot \\text{fun}(\\boldsymbol{\\Phi}:\\mathbf{d},\\mathbf{V})'
# test_pseudocode = 'dot(frac(plus(y_bold,y_bold),div(d_bold)),fun(contract(Phi_bold,d_bold),V_bold))'


max_seq_length_encoder = 512
max_seq_length_decoder = 128

input_text = 'Generate the pseudocode for this mathematical expression written in LaTeX:\n' + test_latex
input_ids = bart_tokenizer(input_text, padding='max_length', max_length=max_seq_length_encoder, return_tensors="pt").input_ids
# print(input_ids)

temp = 1.0
top_p = 0.9
do_sample = True
outputs = custom_full_model.generate(input_ids, do_sample=do_sample, temperature=temp, top_p=top_p, max_new_tokens=max_seq_length_decoder, no_repeat_ngram_size=None)
# outputs = model.generate(input_ids, do_sample=do_sample, temperature=temp, top_p=top_p, max_new_tokens=max_seq_length)
# outputs = model.generate(input_ids, do_sample=do_sample, temperature=temp, top_k=3, max_new_tokens=max_seq_length)
print(outputs)

response = custom_tokenizer.batch_decode(outputs, skip_special_tokens=False)
print('Temp: ', temp, '. Top_p: ', top_p, '. Do_sample: ', do_sample)
print(response)
