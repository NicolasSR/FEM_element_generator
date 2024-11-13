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

terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# SYS_PROMPT = """
# You are an assistant designed to help with a PDE that will be given by the user.
# In order to do so, please carefully follow the next steps, one by one.

# 1. Find if there is any double-dot product within the expression given by the user. Write all of them concisely. Examples:
# - d = double_dot(r,o)
# - s = double_dot(c,t)

# 2. Determine the ranks of the input and output tensors in each of the operations listed above.
# """

SYS_PROMPT = """
You are an assistant designed to help with a PDE that will be given by the user.
Please clearly state each of the the equations given by the user, but simplifying the notation (getting rid of latex expressions)
Write the expressions as pseudocode.
"""

messages = [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": "The given PDE is: $\int_{\Omega}\\boldsymbol{\sigma} : \\nabla \mathbf{v}\ d\Omega = - \int_{\Omega}\mathbf{f}\cdot\mathbf{v}\ d\Omega$. With: $\\boldsymbol{\sigma} = \\textbf{C} : \\boldsymbol{\\varepsilon}$, and $\\boldsymbol{\\varepsilon} = \\frac{1}{2}((\\nabla \\textbf{u})+(\\nabla \\textbf{u})^T)$"},
    # {'role': 'assistant', 'content': 'Here is the list of minimal symbolic objects needed to define this problem:\n\n- \nabla\n- u\n- v\n- f\n- C'},
    # {"role": "user", "content": "Categorize these symbols in the following groups: 'test function', 'trial function', 'other variable', 'operator'"},
    # {'role': 'assistant', 'content': "Here are the symbols categorized:\n\n- v: 'test function'\n- u: 'trial function'\n- f: 'other variable'\n- C: 'other variable'\n- \nabla: 'operator'"},
    # {"role": "user", "content": "For those symbols that are not operators, can you specify if they are scalar fields, vector fields, or other kinds of tensors?"},
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



