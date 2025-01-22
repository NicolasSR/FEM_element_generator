import numpy as np

class VaraibleIdUnifier():

    def __init__(self, tokenizer):
        
        interchangeable_tokens_dict = {
            'var0_':[],
            'var1_':[],
            'var2_':[],
            'var3_':[],
            'var4_':[],
            'coord_':[]
            }

        self.tokenizer_vocab = tokenizer.get_vocab()

        for token_label, token_id in self.tokenizer_vocab.items():
            if token_label[:-1] in interchangeable_tokens_dict.keys():
                interchangeable_tokens_dict[token_label[:-1]].append(token_id)
        
        for intearchangeable_id_list in interchangeable_tokens_dict.values():
            intearchangeable_id_list = intearchangeable_id_list.sort()

        self.interchangeable_tokens_dict = interchangeable_tokens_dict

        self.inverse_interchangeable_tokens_dict = {}
        for token_id in self.tokenizer_vocab.values():
            for group_name, group_ids_list in self.interchangeable_tokens_dict.items():
                if token_id in group_ids_list:
                    self.inverse_interchangeable_tokens_dict[token_id] = group_name
                    break

    def unify_variable_tokens(self, tokens_list):

        token_appearences = {
            'var0_':[],
            'var1_':[],
            'var2_':[],
            'var3_':[],
            'var4_':[],
            'coord_':[]
        }

        tokens_list_unified = []
        
        for token in tokens_list:
            if token in self.inverse_interchangeable_tokens_dict.keys():
                token_group = self.inverse_interchangeable_tokens_dict[token]
                token_appearence_position = np.argwhere(np.array(token_appearences[token_group]) == token).flatten().tolist()
                if len(token_appearence_position) == 1:
                    tokens_list_unified.append(self.interchangeable_tokens_dict[token_group][token_appearence_position[0]])
                elif len(token_appearence_position) == 0:
                    token_appearences[token_group].append(token)
                    tokens_list_unified.append(self.interchangeable_tokens_dict[token_group][len(token_appearences[token_group])-1])
                else:
                    raise ValueError("length of token_appearence_position is not valid")
            else:
                tokens_list_unified.append(token)
        
        return tokens_list_unified

    def unify_variable_tokens_batch(self, tokens_list_batch):
        tokens_list_batch_unified = []
        for tokens_list in tokens_list_batch:
            tokens_list_batch_unified.append(self.unify_variable_tokens(tokens_list))
        return tokens_list_batch_unified
    
    def get_inverse_interchangeable_tokens_dict(self):
        return self.inverse_interchangeable_tokens_dict