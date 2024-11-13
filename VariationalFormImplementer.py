from transformers import pipeline
import torch

model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(task="text-generation",
        model=model_id,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            # "quantization_config": {"load_in_4bit": True},
        },
        device="cuda")

# messages = [
#     {"role": "system", "content": "You are the goalkeeper to a professional soccer team. Your club just won one of the most renowned competitions in that sport"},
#     {"role": "user", "content": "How are you feeling after such success?"},
# ]

# SYS_PROMPT = """You are an assistant for answering questions.
# You are given the extracted parts of a long document and a question. Provide a conversational answer.
# If you don't know the answer, just say "I do not know." Don't make up an answer."""
SYS_PROMPT = """You will be given the variational form (weak form) of a PDE written in LaTex.
I need you to do the following tasks one after the other, using the results from the previous ones:
1. List the minimal set of symbolic objects that you need to define this problem. Make sure to not list a symbol if it can be derived from others. Do not forget any symbol that does not depend on others.
Print only the list of symbols.
For example, for \\nabla v \cdot \\nabla u:
- v
- u
- \\nabla.
Or for example, in \\nabla \cdot b = 0 with b = \\nabla r:
- r
- \\nabla

2. Categorize these symbols in the following groups: 'test function', 'trial function', 'other variable', 'operator'

3. Print: "FINAL OUTPUT" as a string followed by a newline

4. For those symbols that are not operators, can you specify if they are scalar fields, vector fields, or other kinds of tensors? Format the final result as such:
- symbol_1: test function, vector field
- symbol_2: trial function, scalar field
- symbol_3: operator
- symbol_4: other variable, scalar field
- symbol_5: other variable, tensor field
- symbol_6: operator

Give simple names to the symbols, removing latex formatting
"""

"""
4. Take the list from step 3. For those symbols that are 'vector field', write:
- DefineMatrix('symbol_name',nnodes,dim)
And for those that are 'scalar field', write:
- DefineMatrix('symbol_name',nnodes)
Ignore the rest. Only substitute 'symbol_name' for the actual names, leave 'nnodes' and 'dim' as they are.
"""


"""You will be given the variational form (weak form) of a PDE written in LaTex.
I want to implement it in a discretized way, so the test and trial functions are the dot product of some coefficients times
the basis functions.
I need you to list the minimal set of symbolic objects that you need to define this problem. Make sure to not list a symbol if it can be derived from others. And not to forget any symbol.
Print only the list of symbols.
For example, for \\nabla v \cdot \\nabla u:
- v
- u
- \\nabla.
Or for example, in \\nabla \cdot b = 0 with b = \\nabla r:
- r
- \\nabla.
""" 

"""
I need you to list all of the symbols in the given PDE and determine if they are a test function, a trial function or something else.
The format needs to be the following.
For symbols d, v, n, p:
- d: test function
- v: trial funciton
- n: other
- p: trial function

Do not write any other text
"""

messages = [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": "We are solving for a function that describes temperature. The PDE is: \int_{\Omega} \\nabla u_h \cdot \\nabla v {\, \mathrm{d}x} = \int_{\Omega} f v {\, \mathrm{d}x}"},
    # {'role': 'assistant', 'content': '- u_h: trial function\n- v: test function\n- f: other\n- x: other'},
    # {"role": "user", "content": "For the symbols that correspond to test and trial functionsm can you determine if they are scalar fields or vector fields?"},
    # {'role': 'assistant', 'content': '- u_h: trial function\n- v: test function\n- f: other\n- x: other'}, {'role': 'user', 'content': 'For the symbols that correspond to test and trial functionsm can you determine if they are scalar fields or vector fields?'}, {'role': 'assistant', 'content': '- u_h: trial function, vector field\n- v: test function, vector field'},
    # {"role": "user", "content": "For the symbols that correspond 'other', can you determine if they are an operator, the space coordinates or some variable?"},
    ]

# messages = [
#     {"role": "system", "content": SYS_PROMPT},
#     {"role": "user", "content": "\int_{\Omega}\\boldsymbol{\sigma} : \\nabla \mathbf{v}\ d\Omega = - \int_{\Omega}\mathbf{f}\cdot\mathbf{v}\ d\Omega. With: \\boldsymbol{\sigma} = \\textbf{C} : \\boldsymbol{\\varepsilon} and \\boldsymbol{\\varepsilon} = \\frac{1}{2}((\\nabla \\textbf{u})+(\\nabla \\textbf{u})^T)"},
#     # {'role': 'assistant', 'content': 'Here is the list of minimal symbolic objects needed to define this problem:\n\n- \nabla\n- u\n- v\n- f\n- C'},
#     # {"role": "user", "content": "Categorize these symbols in the following groups: 'test function', 'trial function', 'other variable', 'operator'"},
#     # {'role': 'assistant', 'content': "Here are the symbols categorized:\n\n- v: 'test function'\n- u: 'trial function'\n- f: 'other variable'\n- C: 'other variable'\n- \nabla: 'operator'"},
#     # {"role": "user", "content": "For those symbols that are not operators, can you specify if they are scalar fields, vector fields, or other kinds of tensors?"},
#      ]


terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipe(
    messages,
    # max_new_tokens=20000,
    max_new_tokens=2000,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.3,
    top_p=0.9,
)
# assistant_response = outputs[0]["generated_text"][-1]["content"]
assistant_response = outputs[0]
print(assistant_response)

final_resonse = assistant_response['generated_text'][-1]['content']
phrase = "FINAL OUTPUT"

# Split the string at the phrase and get the part after it
result = final_resonse.split(phrase, 1)[1].strip()

print(result)

basic_vars_list = []
for line in result.splitlines():
    name = line.split(':', 1)[0].strip()[2:]
    if 'vector field' in line:
        print(f"DefineMatrix('{name}',nnodes,dim)")
        basic_vars_list.append(name)
    elif 'scalar field' in line:
        print(f"DefineVector('{name}',nnodes)")
        basic_vars_list.append(name)

