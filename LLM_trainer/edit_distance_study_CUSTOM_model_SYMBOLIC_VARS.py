import numpy as np

from datasets import load_from_disk

import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

from custom_decoder_model.custom_tokenizer_test import CustomTokenizer
from variable_id_unifier import VaraibleIdUnifier

def generate_tree(tokens_list):
    tree = nx.DiGraph()

    start_not_cropped = True
    while start_not_cropped:
        if tokens_list[0] in [0,2]:
            tokens_list.pop(0)
        else:
            start_not_cropped = False

    current_node = 0
    tree.add_node(current_node, value=tokens_list[current_node])
    next_edge_side = 'L'
    current_parent = 0
    pending_bifurcations = [0]
    for val in tokens_list[1:]:
        if val == 1:
            continue
        elif val == 2:
            break
        elif val == 32: # Close_parenthesis
            try:
                pending_bifurcations.pop()
            except:
                break
        elif val == 31: # Comma
            try:
                pending_bifurcations.pop()
                current_parent = pending_bifurcations[-1]
                next_edge_side = 'R'
            except:
                break
        else:
            current_node+=1
            tree.add_node(current_node, value=val)
            tree.add_edge(current_parent, current_node, side=next_edge_side)
            current_parent=current_node
            pending_bifurcations.append(current_node)
            next_edge_side = 'L'

    if pending_bifurcations != [0]:
        print('Conversion to graph failed')
        print(tokens_list)
        print(pending_bifurcations)
        tree = None
    return tree

vocab_dict = {
    'vars_all':[i+33 for i in range(50)],
    'coords':[i+83 for i in range(3)]
}

def node_subst_fun(node1, node2):
    if node1['value']==node2['value']:
        return 0.0
    elif node1['value'] in vocab_dict['vars_all'] and node2['value'] in vocab_dict['vars_all']:
        if node2['value']-33 >= (node1['value']-33)//10*10 and node2['value']-33 < (node1['value']-33)//10*10+10:
            return 0.5
        else:
            return 1.0
    elif node1['value'] in vocab_dict['vars_all'] or node2['value'] in vocab_dict['vars_all']:
        return 1.5
    elif node1['value'] in vocab_dict['coords'] and node2['value'] in vocab_dict['coords']:
        return 0.5
    elif node1['value'] in vocab_dict['coords'] or node2['value'] in vocab_dict['coords']:
        return 1.5
    else:
        return 1.0
def node_ins_del_fun(node1):
    return 1.0
def edge_subst_fun(edge1, edge2):
    if edge1['side']==edge2['side']:
        return 0.0
    else:
        return 0.5
def edge_ins_del_fun(edge1):
    return 1.0

custom_tokenizer = CustomTokenizer(vocab_file="LLM_trainer/custom_decoder_model/vocab_symbolic_vars.json")

# dataset_test_noFun = load_from_disk("latex_pseudocode_dataset_validation_1K.hf")
dataset_test = load_from_disk("test_dataset_with_outputs.hf")

# print(dataset_test_noFun)

# print(dataset_test_noFun[4])
# exit()

equivalent_tokens_unifier = VaraibleIdUnifier(custom_tokenizer)

edit_distances_list = []
for i in range(dataset_test.num_rows):
    # if i%100 == 0:
    #     print(i)
    print(i)

    tokens_true = dataset_test['labels'][i]
    tokens_true_unified = equivalent_tokens_unifier.unify_variable_tokens(tokens_true)


    tree_true = generate_tree(tokens_true_unified)
    labels_true = nx.get_node_attributes(tree_true, 'value')
    edge_labels_true = nx.get_edge_attributes(tree_true,'side')

    tokens_guess_list = dataset_test['generation_output_ids'][i]
    tokens_guess_list_unified = equivalent_tokens_unifier.unify_variable_tokens_batch(tokens_guess_list)
    edit_distances_list.append([])
    for j, tokens_guess_unified in enumerate(tokens_guess_list_unified):
        tree_guess = generate_tree(tokens_guess_unified)
        if tree_guess is None:
            edit_distances_list[-1].append(1000.0)
        else:
            labels_guess = nx.get_node_attributes(tree_guess, 'value')
            edge_labels_guess = nx.get_edge_attributes(tree_guess,'side')

            edit_distance = nx.graph_edit_distance(tree_true,tree_guess, roots=(0,0),
                                    node_subst_cost=node_subst_fun,
                                    node_del_cost=node_ins_del_fun, node_ins_cost=node_ins_del_fun,
                                    edge_subst_cost=edge_subst_fun, 
                                    edge_del_cost=edge_ins_del_fun, edge_ins_cost=edge_ins_del_fun)
            edit_distances_list[-1].append(float(edit_distance))


print(len(edit_distances_list))
dataset_test = dataset_test.add_column('edit_distances', edit_distances_list)
print(dataset_test)

dataset_test.save_to_disk("test_dataset_with_edit_distances.hf")
        
        
        # subax1 = plt.subplot(121)
        # pos = graphviz_layout(tree_true, prog="dot")
        # nx.draw(tree_true, pos, labels=labels_true, with_labels=True, font_weight='bold')
        # nx.draw_networkx_edge_labels(tree_true, pos, edge_labels = edge_labels_true)
        # plt.title('Ground truth')
        # subax1 = plt.subplot(122)
        # pos = graphviz_layout(tree_guess, prog="dot")
        # nx.draw(tree_guess, pos, labels=labels_guess, with_labels=True, font_weight='bold')
        # nx.draw_networkx_edge_labels(tree_guess, pos, edge_labels = edge_labels_guess)
        # plt.title(f'Edit distance: {edit_distance}')
        # plt.show()
