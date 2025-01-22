from collections import Counter
import json

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk

from custom_decoder_model.custom_tokenizer_test import CustomTokenizer
from variable_id_unifier import VaraibleIdUnifier

custom_tokenizer = CustomTokenizer(vocab_file="LLM_trainer/custom_decoder_model/vocab_symbolic_vars.json")

variable_id_unifier = VaraibleIdUnifier(custom_tokenizer)

with open('LLM_trainer/custom_decoder_model/vocab_symbolic_vars.json', 'r') as tokens_json_file:
    tokens_json = json.load(tokens_json_file)

tokens_json = {v: k for k, v in tokens_json.items()}
string_tokens_list=[]
for i in range(len(tokens_json)):
    string_tokens_list.append(tokens_json[i])

# ds = load_from_disk("latex_pseudocode_dataset_train_10K.hf")
ds = load_from_disk("test_dataset_with_edit_distances.hf")
print(ds)

""" 
max_seq_length_decoder = 128
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    label_texts = [convo[1]['content'] for convo in convos]

    # model_inputs = bart_tokenizer(input_texts, max_length=max_seq_length_encoder, truncation=True)
    model_labels = custom_tokenizer(label_texts, max_length=max_seq_length_decoder, truncation=True)

    examples["labels"] = model_labels["input_ids"]

    return examples

ds = ds.map(formatting_prompts_func, batched = True,)

len_vec = []
for label in ds['labels']:
    len_vec.append(len(label))
print('Max: ', max(len_vec))
print('Min: ', min(len_vec))
print('Mean: ', np.mean(len_vec))
print('Median: ', np.median(len_vec))
exit()

label_ids_histogram = {}
for i in range(len(ds)):
    id_label = ds[i]['labels']
    id_label_length = len(id_label)

    for id in id_label:
        if id in label_ids_histogram.keys():
            label_ids_histogram[id]+=1
        else:
            label_ids_histogram[id]=1

plt.bar(label_ids_histogram.keys(), label_ids_histogram.values(), color='b')
plt.xticks(list(range(len(string_tokens_list))), labels=string_tokens_list, rotation=75)
plt.show()

exit()
 """

edit_distances_all = ds['edit_distances']

mean_edit_distance_histogram = {}
mean_nodes_difference_size_histogram = {}
node_difference_ids_histogram = {}
label_ids_histogram = {}

# for i in range(10):
for i in range(len(ds)):
    edit_distances_list = edit_distances_all[i]
    mean_edit_distance = np.mean(edit_distances_list)
    # print()
    # print(ds[i]['conversations'][1]['content'])
    # print(mean_edit_distance)
    # print(len(ds[i]['labels']))
    id_outputs = variable_id_unifier.unify_variable_tokens_batch(ds[i]['generation_output_ids'])
    string_outputs = custom_tokenizer.batch_decode(id_outputs, skip_special_tokens=True)
    id_label = variable_id_unifier.unify_variable_tokens(ds[i]['labels'])
    id_label_length = len(id_label)
    # for j in range(len(string_outputs)):
    #     print(edit_distances_list[j], ': ', string_outputs[i])

    if 1000 not in edit_distances_list:
    # if True:
        for id in id_label:
            if id in label_ids_histogram.keys():
                label_ids_histogram[id]+=1
            else:
                label_ids_histogram[id]=1

        for j in range(len(id_outputs)):
            intersected_ids_list = [i for i in list((Counter(id_label) & Counter(id_outputs[j])).elements()) if i not in [0,1,2, 31, 32]]
            union_ids_list = [i for i in list((Counter(id_label) | Counter(id_outputs[j])).elements()) if i not in [0,1,2, 31, 32]]
            difference_ids_set = set(union_ids_list).difference(set(intersected_ids_list))
            for id in difference_ids_set:
                if id in node_difference_ids_histogram.keys():
                    node_difference_ids_histogram[id]+=1
                else:
                    node_difference_ids_histogram[id]=1
            nodes_difference_size = len(union_ids_list)-len(intersected_ids_list)
            # jaccard_similarities = 1-len(intersected_ids_list)/len(union_ids_list)
        mean_nodes_difference_size = np.mean(nodes_difference_size)

        if id_label_length in mean_edit_distance_histogram.keys():
            mean_edit_distance_histogram[id_label_length].append(mean_edit_distance)
            mean_nodes_difference_size_histogram[id_label_length].append(mean_nodes_difference_size)
        else:
            mean_edit_distance_histogram[id_label_length]=[mean_edit_distance]
            mean_nodes_difference_size_histogram[id_label_length]=[mean_nodes_difference_size]

labels, data = [*zip(*mean_edit_distance_histogram.items())]  # 'transpose' items to parallel key, value lists

print('Non-catastrophic ratio: ', sum([len(a) for  a in data])/10.0, '%')

print(node_difference_ids_histogram)
print(label_ids_histogram)

fig, ax1 = plt.subplots()
color = 'black'
ax1.set_xlabel('sequence length (tokens)')
ax1.set_ylabel('mean edit distance distribution', color=color)
ax1.boxplot(data, positions=labels)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('number of samples', color=color)  # we already handled the x-label with ax1
ax2.scatter(labels, [len(a) for  a in data], marker='x', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


labels, data = [*zip(*mean_nodes_difference_size_histogram.items())]  # 'transpose' items to parallel key, value lists

fig, ax1 = plt.subplots()
color = 'black'
ax1.set_xlabel('sequence length (tokens)')
ax1.set_ylabel('mean nodes difference size distribution', color=color)
ax1.boxplot(data, positions=labels)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('number of samples', color=color)  # we already handled the x-label with ax1
ax2.scatter(labels, [len(a) for  a in data], marker='x', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


plt.bar([val-0.2 for val in label_ids_histogram.keys()], label_ids_histogram.values(), 0.4, color='b', label='In ground truth')
plt.bar([val+0.2 for val in node_difference_ids_histogram.keys()], node_difference_ids_histogram.values(), 0.4, color='r', label='Mistaken')
plt.xticks(list(range(len(string_tokens_list))), labels=string_tokens_list, rotation=75)
plt.xlabel('token label')
plt.ylabel('occurences')
plt.legend()
plt.tight_layout()
plt.show()