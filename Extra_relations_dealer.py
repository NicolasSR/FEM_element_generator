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

SYS_PROMPT = """
You are an assistant designed to help write SymPy code for a specific PDE that will be given by the user.
In order to do so, please carefully follow the next steps, one by one.

1. Concisely list the extra relations specified to complete the PDE (if there are any)

2. Write the pseudocode for each of them, having already defined the varibles:
    u_vec (trial function, vector field)
    v_vec (test function, vector field)
    
    And also the funtions:
    grad.sym(x) -> Computes the symmetric part of the gradient of field x and gives the result in voigt notation.
    define_symmetric_matrix(y,z): -> Defines a symmetric matrix of size (y,z)
    do_voigt -> Recieves a matrix in standard notation and returns it in Voigt notation
    grad(x) -> Returns the gradient of a given scalar or vector field x
    sym_part(x) -> Takes matrix x and computes 1/2(x+x^T)
    undo_voigt(p) -> Recieves a matrix in Voigt notation and returns it in standard notation
    dot_product(f,g) -> Computes the dot product of matrix f by 
"""

messages = [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": "The given PDE is: \int_{\Omega}\\boldsymbol{\sigma} : \\nabla \mathbf{v}\ d\Omega = - \int_{\Omega}\mathbf{f}\cdot\mathbf{v}\ d\Omega. With: \\boldsymbol{\sigma} = \\textbf{C} : \\boldsymbol{\\varepsilon} and \\boldsymbol{\\varepsilon} = \\frac{1}{2}((\\nabla \\textbf{u})+(\\nabla \\textbf{u})^T)"+
     "Explain what the : operator means, and what symmetries do we find in tensor C. Then, do the tasks specified at the beginning."},
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



