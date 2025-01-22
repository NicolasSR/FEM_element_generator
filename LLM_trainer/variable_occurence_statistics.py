import json

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk

from custom_decoder_model.custom_tokenizer_test import CustomTokenizer
from variable_id_unifier import VaraibleIdUnifier

custom_tokenizer = CustomTokenizer(vocab_file="LLM_trainer/custom_decoder_model/vocab_symbolic_vars.json")

variable_id_unifier = VaraibleIdUnifier(custom_tokenizer)

# ds = load_from_disk("latex_pseudocode_dataset_train_10K.hf")
ds = load_from_disk("latex_pseudocode_dataset_train_10K.hf")
print(ds)

string_labels = [convo[1]['content'] for convo in ds['conversations']]
id_labels = custom_tokenizer(string_labels)['input_ids']

occurences_dict = {
    "var0_": {
        "pending": [0 for i in range(10)],
        "identified": [0 for i in range(10)],
        "occ_rank": [0 for i in range(10)],
        # "appeared": []
    },
    "var1_": {
        "pending": [0 for i in range(10)],
        "identified": [0 for i in range(10)],
        "occ_rank": [0 for i in range(10)],
        # "appeared": []
    },
    "var2_": {
        "pending": [0 for i in range(10)],
        "identified": [0 for i in range(10)],
        "occ_rank": [0 for i in range(10)],
        # "appeared": []
    },
    "var3_": {
        "pending": [0 for i in range(10)],
        "identified": [0 for i in range(10)],
        "occ_rank": [0 for i in range(10)],
        # "appeared": []
    },
    "var4_": {
        "pending": [0 for i in range(10)],
        "identified": [0 for i in range(10)],
        "occ_rank": [0 for i in range(10)],
        # "appeared": []
    },
    "coord_": {
        "pending": [0 for i in range(3)],
        "identified": [0 for i in range(3)],
        "occ_rank": [0 for i in range(10)],
        # "appeared": []
    },
}

inverse_tokens_dict = variable_id_unifier.get_inverse_interchangeable_tokens_dict()

for id_label in id_labels:
    for group_info in occurences_dict.values():
        group_info["appeared"] = []
        group_info["identified_local"] = [0 for i in range(len(group_info["identified"]))]
    for id in id_label:
        if id in inverse_tokens_dict.keys():
            group_info = occurences_dict[inverse_tokens_dict[id]]
            if id in group_info["appeared"]:
                group_info['identified_local'][group_info["appeared"].index(id)]+=1
                group_info['identified'][group_info["appeared"].index(id)]+=1
            else:
                group_info['pending'][len(group_info["appeared"])]+=1
                group_info["appeared"].append(id)
    for group_info in occurences_dict.values():
        for i in range(len(group_info["occ_rank"])):
            group_info["occ_rank"][i]+=len(np.where(np.array(group_info['identified_local'])>i)[0].flatten().tolist())

print(occurences_dict)

for i, (group_name, group_info) in enumerate(occurences_dict.items()):
    plt.bar([j+(0.1*i)-0.3 for j in range(len(group_info["pending"]))], group_info["pending"], 0.1, label=group_name)
    plt.title("Pending")
    plt.xlabel("occurences")
    plt.ylabel("Variable index")
    plt.ylim([0, 6000])
    plt.legend()
plt.show()

for i, (group_name, group_info) in enumerate(occurences_dict.items()):
    plt.bar([j+(0.1*i)-0.3 for j in range(len(group_info["identified"]))], group_info["identified"], 0.1, label=group_name)
    plt.title("Identified")
    plt.xlabel("occurences")
    plt.ylabel("Variable index")
    plt.ylim([0, 6000])
    plt.legend()
plt.show()

for i, (group_name, group_info) in enumerate(occurences_dict.items()):
    plt.bar([j+(0.1*i)-0.3 for j in range(len(group_info["occ_rank"]))], group_info["occ_rank"], 0.1, label=group_name)
    plt.title("Occurence rank")
    plt.xlabel("occurences")
    plt.ylabel("Variable index")
    plt.ylim([0, 6000])
    plt.legend()
plt.show()