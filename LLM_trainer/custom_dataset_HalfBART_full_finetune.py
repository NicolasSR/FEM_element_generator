import numpy as np

import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from LLM_trainer.custom_decoder_model.custom_model_configuration_explicit_vars import CustomConfig
from custom_decoder_model.custom_tokenizer_test import CustomTokenizer
from custom_decoder_model.custom_decoder_model_test import CustomDecoder

bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
bart_full_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
# bart_full_config = AutoConfig.from_pretrained("facebook/bart-base")
# bart_full_model = AutoModelForSeq2SeqLM.from_config(bart_full_config)

bart_encoder =bart_full_model.model.encoder
bart_decoder =bart_full_model.model.decoder

test_latex = '\\langle \\boldsymbol{\\tau} \\otimes \\mathbf{V},\\boldsymbol{\\Upsilon} \\rangle \\cdot \\sgn{\\langle \\mathbf{C},\\mathbf{V} \\rangle}'
test_pseudocode = 'prod(inner(outer(tau_bold,V_bold),Upsilon_bold),sign(inner(C_bold,V_bold)))'


input_ids = bart_tokenizer.encode(test_latex, return_tensors='pt')
print(input_ids)
# encoder_embeddings = bart_encoder(input_ids)['last_hidden_state']
encoder_outputs = bart_encoder(input_ids)
print(encoder_outputs)


# # model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)
# out_gen = bart_full_model.generate(input_ids, encoder_outputs=encoder_outputs)
# print(out_gen)
# out_gen_string = bart_tokenizer.batch_decode(out_gen)
# print(out_gen_string)
# exit()

custom_tokenizer  = CustomTokenizer(vocab_file="LLM_trainer/custom_decoder_model/vocab.json")
custom_config = CustomConfig.from_pretrained('LLM_trainer/custom_decoder_model/')
custom_model = CustomDecoder._from_config(custom_config)

# decoder_outputs = bart_encoder(
#     input_ids=input_ids,
#     attention_mask=attention_mask,
#     head_mask=head_mask,
#     inputs_embeds=inputs_embeds,
#     output_attentions=output_attentions,
#     output_hidden_states=output_hidden_states,
#     return_dict=return_dict,
# )


custom_model.generate(input_ids=None, tokenizer=custom_tokenizer, encoder_outputs=encoder_outputs)
# custom_model.generate(input_ids=None, tokenizer=custom_tokenizer)
exit()

decoder_outputs = custom_model(
            input_ids=torch.tensor([[0]]),
            encoder_hidden_states=encoder_outputs['last_hidden_state']
        )
print(decoder_outputs['logits'].shape)
print(np.argmax(decoder_outputs['logits'].detach().numpy()))



exit()