from transformers import pipeline
import torch

model_id = "meta-llama/Llama-3.1-8B-Instruct"
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
SYS_PROMPT = """You will be given a problem related to Finite Emelent Method for any kind of physics problem. I need you to focus on the dimensions of the domain at hand, and specify it in the following format:
domain_dimensions = [2,3]
or
domain_dimensions = [1]
for example
Output just that line of code, nothing else"""

messages = [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": "We are simulating a Navier-Stokes problem for the behavior of the blood within the model of a real heart."},
    # {"role": "assistant", "content": "I'm ready to help. Please go ahead and provide the extract from the paper about the Finite Elements Method, and I'll do my best to assist you with the mathematical expressions."},
    ]


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
    temperature=0.6,
    top_p=0.9,
)
# assistant_response = outputs[0]["generated_text"][-1]["content"]
assistant_response = outputs[0]
print(assistant_response)