from typing import Optional, List
import re

import numpy as np
from datasets import Dataset

# from bigtree import list_to_tree, tree_to_dot

from seeded_random_generator import SeededRandomGenerator as Utilities
# from Random_PDE_generator.latex_parser_old import LatexParser, conversions_list, parenthesis_table, categories_map
from operators_dict_handler_nofun import OperatorDictHandler, operators_dict, parenthesis_table, precedence_map
from variables_dict_handler import VariableDictHandler, variables_dict
from coords_dict_handler import CoordsDictHandler, coords_dict


class OperatorsTree():

    def __init__(self, operators_dict_handler: OperatorDictHandler, variables_dict_handler: VariableDictHandler,
                 coords_dict_handler: CoordsDictHandler, first_operator, manual_rank:Optional[int]=None):
        # print('TREE INITIALIZED')
        self.operators_dict_handler = operators_dict_handler
        self.variables_dict_handler = variables_dict_handler
        self.coords_dict_handler = coords_dict_handler
        first_node_class = self.operators_dict_handler.get_class_from_operator(first_operator)
        self.nodes_list = first_node_class.create_new_node_as_root(0, self, first_operator, manual_rank)
        self.root_node = self.node(0)
        self.leaf_nodes_list = None
        self.upwards_traversal_order = None

        self.variables_dict={0:{}, 1:{}, 2:{}, 3:{}, 4:{}}

        self.tree_latex_string = None

    def node(self, id):
        return self.nodes_list[id]

    def get_random_new_node(self):

        candidate_nodes = []
        for node in self.nodes_list:
            if node.has_free_spot():
                candidate_nodes.append(node.node_id)
        chosen_associated_node = self.node(Utilities.get_random_element(candidate_nodes)[0])
        child_operator_candidates, chosen_child_slot = chosen_associated_node.get_child_candidates()
        chosen_child_operator = Utilities.get_random_element(child_operator_candidates)[0]
        chosen_child_node_class = self.operators_dict_handler.get_class_from_operator(chosen_child_operator)
        new_node_id = len(self.nodes_list)
        new_child_nodes = chosen_child_node_class.create_new_node_as_child(new_node_id, chosen_child_operator,
                                                                           chosen_associated_node, chosen_child_slot)
        self.nodes_list.extend(new_child_nodes)

    def fill_variable_slots(self):
        node_id = len(self.nodes_list)
        var_nodes = []
        for node in self.nodes_list:
            new_nodes = node.create_variable_nodes(node_id)
            var_nodes.extend(new_nodes)
            node_id+=len(new_nodes)
        self.nodes_list.extend(var_nodes)

    def get_upwards_traversal_order(self):
        self.leaf_nodes_list = []

        nodes_id_set = set([node.node_id for node in self.nodes_list])
        bifurcations = []
        stack=[]
        output_list = []

        current_node = self.root_node

        while len(nodes_id_set) > 0:
            stack.append(current_node.node_id)
            nodes_id_set.remove(current_node.node_id)
            if current_node.get_traversal_node_type() == "leaf":
                self.leaf_nodes_list.append(current_node.node_id)
                bifurcation_found = False
                while bifurcation_found == False  and len(stack)>0:
                    if stack[-1] in bifurcations:
                        next_node_id = self.node(stack[-1]).right_child_id
                        bifurcations.pop()
                        bifurcation_found = True
                    else:
                        output_list.append(stack.pop())
            elif current_node.get_traversal_node_type() == "bifurcation":
                bifurcations.append(current_node.node_id)
                next_node_id = current_node.left_child_id
            elif current_node.get_traversal_node_type() == "pass":
                next_node_id = current_node.left_child_id
            else:
                raise ValueError(f"Uncompatible value for traversal_node_type: {current_node.get_traversal_node_type()}")
            
            current_node = self.node(next_node_id)

        self.upwards_traversal_order = output_list
    
    # def visualize_tree(self):
    #     all_paths=[]
        
    #     for leaf_id in self.leaf_nodes_list:
    #         current_node = self.node(leaf_id)
    #         current_str=str(current_node.out_rank)+' ['+current_node.grammar+','+str(current_node.node_id)+']'
    #         while current_node != self.root_node:
    #             current_parent = self.node(current_node.parent_id)
    #             current_str = str(current_parent.out_rank)+' ['+current_parent.grammar+','+str(current_parent.node_id)+']/'+current_str
    #             current_node = current_parent
    #         all_paths.append(current_str)

    #     graph = tree_to_dot(list_to_tree(all_paths), node_colour="white")
    #     graph.write_png("tree.png")

    def generate_latex_string(self):

        for node_id in self.upwards_traversal_order:
            node = self.node(node_id)
            node.generate_latex_string()
        assert node == self.root_node, "The last node at the traversal order is supposed to be rhe root, but somehow is not in this case"

        self.tree_latex_string = self.root_node.latex_string

    def generate_pseudocode_string(self):

        for node_id in self.upwards_traversal_order:
            node = self.node(node_id)
            node.generate_pseudocode_string()
        assert node == self.root_node, "The last node at the traversal order is supposed to be rhe root, but somehow is not in this case"

        self.tree_pseudocode_string = self.root_node.pseudocode_string


    

class PDEWeakFormGenerator():

    def __init__(self, operators_dict_handler, variables_dict_handler, coords_dict_handler):
        self.operators_dict_handler = operators_dict_handler
        self.variables_dict_handler = variables_dict_handler
        self.coords_dict_handler = coords_dict_handler
    
    def generate_expression(self, iterations_range):
        self.operators_dict_handler.reset()
        self.variables_dict_handler.reset()
        self.coords_dict_handler.reset()

        root_operator_candidates = self.operators_dict_handler.list_all_operators()
        # root_operator_candidates = ['prod(a,b)', 'dot(a,b)', 'inner(a,b)']
        root_operator = Utilities.get_random_element(root_operator_candidates)[0]

        # manual_rank = 0
        rank_candidates = self.operators_dict_handler.get_compatible_out_ranks(root_operator)
        manual_rank = Utilities.get_random_element(rank_candidates)[0]

        tree = OperatorsTree(operators_dict_handler, variables_dict_handler, coords_dict_handler, root_operator, manual_rank)

        iterations = Utilities.get_random_element(list(range(iterations_range[0],iterations_range[1]+1)))[0]
        for iter in range(iterations):
            tree.get_random_new_node()

        tree.fill_variable_slots()
        tree.get_upwards_traversal_order()
        # tree.visualize_tree()
        tree.generate_latex_string()
        tree.generate_pseudocode_string()

        return tree

if __name__=="__main__":


    import matplotlib.pyplot as plt

    # seed = np.random.randint(100000)
    seed = 1234
    # seed = 0 # For training set
    # seed = 1 # For validation set
    # seed = 2 # For test set
    Utilities.set_seed(seed)

    operators_dict_handler = OperatorDictHandler(operators_dict, parenthesis_table, precedence_map)
    variables_dict_handler = VariableDictHandler(variables_dict)
    coords_dict_handler = CoordsDictHandler(coords_dict)

    pde_generator = PDEWeakFormGenerator(operators_dict_handler, variables_dict_handler, coords_dict_handler)

    """ 
    ds=None

    for i in range(1000):
        if i%1000==0:
            print(i)

        tree = pde_generator.generate_expression([0,6])
        latex_string = tree.tree_latex_string
        pseudocode_string = tree.tree_pseudocode_string

        formatted_latex_string = re.sub(r':/:', r'\\\\', latex_string)

        if ds is None:
            ds = Dataset.from_dict({"conversations": [[{'role': 'user', 'content': formatted_latex_string},{'role': 'assistant', 'content': pseudocode_string}]], "seed": [seed]})
        else:
            ds = ds.add_item({"conversations": [{'role': 'user', 'content': formatted_latex_string},{'role': 'assistant', 'content': pseudocode_string}], "seed": seed})

    ds.save_to_disk("test_dataset.hf")
    """

    
    for i in range(10):

        tree = pde_generator.generate_expression([0,6])
        latex_string = tree.tree_latex_string
        pseudocode_string = tree.tree_pseudocode_string

        print("test_latex = '"+re.sub(r':/:', r'\\\\', latex_string)+"'")
        print("test_pseudocode = '"+pseudocode_string+"'")
        # print('$'+re.sub(r':/:', r'\\', latex_string)+'$\\\\')
        # print(re.sub(r'_', r'\\_', pseudocode_string)+'\\\\')
        # print()
   

    # full_latex_string = re.sub(':/:', r'\\', operators_tree.tree_latex_string)
    # print(full_latex_string)

    # latex_expression = '$'+full_latex_string+'$'
    # fig = plt.figure(figsize=(10, 10))  # Dimensions of figsize are in inches
    # text = fig.text(
    #     x=0.5,  # x-coordinate to place the text
    #     y=0.5,  # y-coordinate to place the text
    #     s=latex_expression,
    #     horizontalalignment="center",
    #     verticalalignment="center",
    #     fontsize=32,
    # )
    # plt.savefig('out_latex.png')
    # plt.close()
   