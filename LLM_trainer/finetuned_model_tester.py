import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from evaluate import load as eval_load


modelpath = "Llama_3.2_TestFineTunedCheckpoint_seed3407"
# modelpath = "TestFineTunedCheckpoint_seed3408"
model = AutoModelForCausalLM.from_pretrained(modelpath)
tokenizer = AutoTokenizer.from_pretrained(modelpath)

max_seq_length = 512 # 512 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

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

test_latex = '\\frac{\\mathbf{y}+\\mathbf{y}}{\\nabla \\cdot \\mathbf{d}} \\cdot \\text{fun}(\\boldsymbol{\\Phi}:\\mathbf{d},\\mathbf{V})'
test_pseudocode = 'dot(frac(plus(y_bold,y_bold),div(d_bold)),fun(contract(Phi_bold,d_bold),V_bold))'

convo = [{'role': 'user', 'content': test_latex}]
prompt = tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# print(input_ids)

temp = 1.0
top_p = 0.9
do_sample = True
outputs = model.generate(input_ids, do_sample=do_sample, temperature=temp, top_p=top_p, max_length=max_seq_length, eos_token_id=128009, pad_token_id=128009, no_repeat_ngram_size=None)
# print(outputs)

response = tokenizer.batch_decode(outputs, skip_special_tokens=False)
print('Temp: ', temp, '. Top_p: ', top_p, '. Do_sample: ', do_sample)
print(response)

references = [['<|begin_of_text|><|begin_of_text|><|start_header_id|>system<|end_header_id|>Generate the pseudocode for this mathematical expression written in LaTeX:<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\\langle \\boldsymbol{\\sigma} \\otimes \\mathbf{W},\\boldsymbol{\\Upsilon} \\rangle \\cdot \\sgn{\\langle \\mathbf{C},\\mathbf{W} \\rangle}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>prod(inner(outer(sigma_bold,W_bold),Upsilon_bold),sign(inner(C_bold,W_bold)))<|eot_id|>']]
bleu = eval_load("bleu")
results = bleu.compute(predictions=response, references=references)
print(results)

