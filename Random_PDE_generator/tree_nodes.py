from typing import List
import re

import numpy as np

from seeded_random_generator import SeededRandomGenerator as Utilities
# from UFL_Grammar_Variational_new import OperatorsTree

class TreeNode():

    # def __init__(self, node_id: int, tree: OperatorsTree):
    def __init__(self, node_id: int, tree):

        self.node_id = node_id
        self.tree = tree
        self.parent_id = None
        self.out_rank = None
        self.traversal_node_type = None
        self.latex_string = None
        self.precedence_group = None
        self.pseudocode_string = None
        
    def set_parent(self, parent_node, child_slot, node_id):
        self.parent_id = parent_node.node_id
        self.out_rank = parent_node.get_child_rank(child_slot)
        parent_node.set_child_id(node_id, child_slot)

    def _replace_latex_string(self, raw_latex, in_latex, slot_precedence_group, in_precedence_group, placeholder):
        
        parenthesis_table = self.tree.operators_dict_handler.parenthesis_table
        precedence_map = self.tree.operators_dict_handler.precedence_map

        parenthesis_notations = (('(',')'),(':/:left (',':/:right )'),('[',']'),(':/:left [',':/:right ]'))
        needs_parenthesis = parenthesis_table[precedence_map["in_type"][in_precedence_group],precedence_map["slot_type"][slot_precedence_group]]
        repl=in_latex

        apply_parenthesis = False
        if needs_parenthesis == 1:
            apply_parenthesis = True
        elif needs_parenthesis == 2:
            apply_parenthesis = self._custom_parenthesis_check(slot_precedence_group, in_precedence_group)
        elif in_precedence_group!="punctuation": 
            apply_parenthesis = Utilities.get_chance(0.1)
        
        if apply_parenthesis:
            parenthesis_type = Utilities.get_random_choice([0.4,0.4,0.1,0.1])
            repl = parenthesis_notations[parenthesis_type][0]+repl+parenthesis_notations[parenthesis_type][1]
            out_string = re.sub(placeholder, repl, raw_latex)
            out_string = re.sub(':_:', ' ', out_string)
        else:
            out_string = re.sub(placeholder, repl, raw_latex)
            out_string = re.sub(':_:', '\ ', out_string)
            
        return out_string
    
    def _replace_pseudocode_string(self, current_string, in_pseudocode, placeholder):
        return re.sub(placeholder, in_pseudocode, current_string)

class UnaryOperatorNode(TreeNode):

    left_placeholder = r"\ba\b"
    compatible_children_groups = ("unary", "binary", "partial_unary", "poly")

    def __init__(self, node_id, tree, operator:str):
        super().__init__(node_id, tree)
        self.grammar = operator

        self.left_child_id = None
        self.left_child_rank = None

        operator_info = self.tree.operators_dict_handler.get_info_from_operator(operator)
        try:
            self.output_rank_relation: str = operator_info["out_relation"]
            self.input_rank_possibilities: List[int] = operator_info["in_ranks"]
            self.output_rank_possibilities: List[int] = operator_info["out_ranks"]
        except:
            raise AttributeError(f"Incorrect typing in the entries of operator info list for {operator}")
        
    @classmethod
    def get_rank_table(cls, grammar_dict):
        rank_table=np.zeros((len(grammar_dict),5),dtype=int)
        for i, value in enumerate(grammar_dict.values()):
            rank_table[i,value["out_ranks"]]=1
        return rank_table

    @classmethod
    def create_new_node_as_child(cls, node_id, operator, parent_node, child_slot):
        new_node = cls(node_id, parent_node.tree, operator) # Instantiate the class
        new_node.set_parent(parent_node, child_slot, node_id)
        new_node.set_node_ranks()
        return [new_node]
    
    @classmethod
    def create_new_node_as_root(cls, node_id, tree, operator, manual_rank):
        new_node = cls(node_id, tree, operator) # Instantiate the class
        new_node.set_first_node_ranks(manual_rank)
        return [new_node]

    def get_child_rank(self, child_slot):
        assert child_slot == 0, f"Child slot {child_slot} does not apply to UnaryOperatorNode"
        return self.left_child_rank
    
    def set_child_id(self, node_id, child_slot):
        assert child_slot == 0, f"Child slot {child_slot} does not apply to UnaryOperatorNode"
        self.left_child_id = node_id
    
    def set_first_node_ranks(self, manual_rank):
        if manual_rank is None:
            self.out_rank = Utilities.get_random_element(self.output_rank_possibilities)[0]
        else:
            self.out_rank = manual_rank
        self.set_node_ranks()

    def set_node_ranks(self):
        if self.out_rank is not None and self.left_child_rank is None:
            rank_calc_string = self.output_rank_relation+f"=={self.out_rank}"
            for trial_rank in self.input_rank_possibilities:
                if eval(re.sub(self.left_placeholder, str(trial_rank), rank_calc_string)):
                    self.left_child_rank = trial_rank
                    break
            assert self.left_child_rank is not None, f"No compatible child rank was found for operator {self.grammar} with out_rank {self.out_rank}"

        else:
            raise ValueError(f"Calling set_node_ranks with out_rank={self.out_rank} and left_child_rank={self.left_child_rank} is not possible for Unary Operator Node.")

    def has_free_spot(self):
        return self.left_child_id is None

    def get_child_candidates(self):
        child_slot = 0
        child_candidates = self.tree.operators_dict_handler.get_compatible_operators_by_rank(self.left_child_rank, self.compatible_children_groups)
        return child_candidates, child_slot
    
    def get_traversal_node_type(self):
        return "pass"
    
    def create_variable_nodes(self, node_id):
        new_nodes_list = []
        if self.left_child_id is None:
            new_node = VariableNode(node_id, self.tree)
            new_node.set_parent(self, 0, node_id)
            new_node.get_var_id()
            new_nodes_list.append(new_node)
        return new_nodes_list
    
    def generate_latex_string(self):
        raw_latex, raw_precedence_group = self.tree.operators_dict_handler.get_operator_latex(self.grammar)
        left_latex = self.tree.node(self.left_child_id).latex_string
        left_precedence_group = self.tree.node(self.left_child_id).precedence_group
        
        slot_precedence_group = raw_precedence_group
        if slot_precedence_group[-2:] not in ('_L','_R'):
            slot_precedence_group += '_L'

        self.latex_string = self._replace_latex_string(raw_latex, left_latex, slot_precedence_group, left_precedence_group, self.left_placeholder)

        self.precedence_group = raw_precedence_group
    
    def generate_pseudocode_string(self):
        raw_pseudocode = self.grammar
        left_pseudocode = self.tree.node(self.left_child_id).pseudocode_string
        self.pseudocode_string = self._replace_pseudocode_string(raw_pseudocode, left_pseudocode, self.left_placeholder)


class BinaryOperatorNode(TreeNode):

    left_placeholder = r"\ba\b"
    right_placeholder = r"\bb\b"
    out_placeholder = r"\bc\b"
    compatible_children_groups = ("unary", "binary", "partial_unary", "poly")

    def __init__(self, node_id, tree, operator:str):
        super().__init__(node_id, tree)
        self.grammar = operator

        self.left_child_id = None
        self.left_child_rank = None
        self.right_child_id = None
        self.right_child_rank = None

        operator_info = self.tree.operators_dict_handler.get_info_from_operator(operator)
        try:
            self.input_rank_relation: str = operator_info["in_relation"]
            self.output_rank_relation: str = operator_info["out_relation"]
            self.input_rank_possibilities: List[List[int]] = operator_info["in_ranks"]
            self.output_rank_possibilities: List[int] = operator_info["out_ranks"]
        except:
            raise AttributeError(f"Incorrect typing in the entries of operator info list for {operator}")
    
    @classmethod
    def get_rank_table(cls, grammar_dict):
        rank_table=np.zeros((len(grammar_dict),5),dtype=int)
        for i, value in enumerate(grammar_dict.values()):
            rank_table[i,value["out_ranks"]]=1
        return rank_table

    @classmethod
    def create_new_node_as_child(cls, node_id, operator, parent_node, child_slot):
        new_node = cls(node_id, parent_node.tree, operator) # Instantiate the class
        new_node.set_parent(parent_node, child_slot, node_id)
        new_node.set_node_ranks()
        return [new_node]
    
    @classmethod
    def create_new_node_as_root(cls, node_id, tree, operator, manual_rank):
        new_node = cls(node_id, tree, operator) # Instantiate the class
        new_node.set_first_node_ranks(manual_rank)
        return [new_node]
    
    def get_child_rank(self, child_slot):
        if child_slot == 0:
            return self.left_child_rank
        elif child_slot == 1:
            return self.right_child_rank
        else:
            raise ValueError(f"Child slot {child_slot} does not apply to BinaryOperatorNode")
    
    def set_child_id(self, node_id, child_slot):
        if child_slot == 0:
            self.left_child_id = node_id
        elif child_slot == 1:
            self.right_child_id = node_id
        else:
            raise ValueError(f"Child slot {child_slot} does not apply to BinaryOperatorNode")

    def set_first_node_ranks(self, manual_rank):
        if manual_rank is None:
            self.out_rank = Utilities.get_random_element(self.output_rank_possibilities)[0]
        else:
            self.out_rank = manual_rank
        self.set_node_ranks()

    def set_node_ranks(self):

        def choose_rank_combination(possibilities_1, possibilities_2, given_rank, placeholder_1, placeholder_2, placeholder_3):
            combinations_grid = np.meshgrid(possibilities_1, possibilities_2)
            combinations = np.hstack([combinations_grid[0].reshape((-1,1)),combinations_grid[1].reshape((-1,1))])
            valid_combinations = []
            rank_calc_string_output = self.output_rank_relation+"==c"
            rank_calc_string_inputs = self.input_rank_relation
            for combo in combinations:
                rank_calc_string_output_sub = re.sub(placeholder_1, str(combo[0]), rank_calc_string_output)
                rank_calc_string_output_sub = re.sub(placeholder_2, str(combo[1]), rank_calc_string_output_sub)
                rank_calc_string_output_sub = re.sub(placeholder_3, str(given_rank), rank_calc_string_output_sub)
                rank_calc_string_inputs_sub = re.sub(placeholder_1, str(combo[0]), rank_calc_string_inputs)
                rank_calc_string_inputs_sub = re.sub(placeholder_2, str(combo[1]), rank_calc_string_inputs_sub)
                rank_calc_string_inputs_sub = re.sub(placeholder_3, str(given_rank), rank_calc_string_inputs_sub)
                if eval(rank_calc_string_output_sub) and eval(rank_calc_string_inputs_sub):
                    valid_combinations.append(combo)
            return list(Utilities.get_random_element(valid_combinations)[0])
        
        if self.out_rank is not None and self.left_child_rank is None and self.right_child_rank is None:
            possibilities_1 = self.input_rank_possibilities[0]
            possibilities_2 = self.input_rank_possibilities[1]
            given_rank = self.out_rank
            placeholder_1 = self.left_placeholder
            placeholder_2 = self.right_placeholder
            placeholder_3 = self.out_placeholder
            chosen_ranks = choose_rank_combination(possibilities_1, possibilities_2, given_rank, placeholder_1, placeholder_2, placeholder_3)
            self.left_child_rank, self.right_child_rank = chosen_ranks
        else:
            raise ValueError("Invalid combination of rank values when calling set_node_ranks in Binary Operator: out_rank={self.out_rank}, left_child_rank={self.left_child_rank}, right_child_rank={self.right_child_rank}.")

    def has_free_spot(self):
        return self.left_child_id is None or self.right_child_id is None

    def get_child_candidates(self):

        if self.left_child_id is None and self.right_child_id is None:
            if Utilities.get_chance(0.5):
                child_slot = 0
                child_rank = self.left_child_rank
            else:
                child_slot = 1
                child_rank = self.right_child_rank
        elif self.left_child_id is None:
            child_slot = 0
            child_rank = self.left_child_rank
        elif self.right_child_id is None:
            child_slot = 1
            child_rank = self.right_child_rank
        else:
            raise ValueError("Either left or right child ID whould be None in order to get a child node in BinaryOperatorNode")
            
        child_candidates = self.tree.operators_dict_handler.get_compatible_operators_by_rank(child_rank, self.compatible_children_groups)
        
        return child_candidates, child_slot
    
    def get_traversal_node_type(self):
        return "bifurcation"
        
    def create_variable_nodes(self, node_id):
        new_nodes_list = []
        if self.left_child_id is None:
            new_node = VariableNode(node_id, self.tree)
            new_node.set_parent(self, 0, node_id)
            new_node.get_var_id()
            new_nodes_list.append(new_node)
            node_id+=1
        if self.right_child_id is None:
            new_node = VariableNode(node_id, self.tree)
            new_node.set_parent(self, 1, node_id)
            new_node.get_var_id()
            new_nodes_list.append(new_node)
        return new_nodes_list
    
    def generate_latex_string(self):
        raw_latex, raw_precedence_group = self.tree.operators_dict_handler.get_operator_latex(self.grammar)

        # Left child:
        left_latex = self.tree.node(self.left_child_id).latex_string
        left_precedence_group = self.tree.node(self.left_child_id).precedence_group
        slot_precedence_group = raw_precedence_group+'_L'

        latex_string = self._replace_latex_string(raw_latex, left_latex, slot_precedence_group, left_precedence_group, self.left_placeholder)

        # Right child:
        right_latex = self.tree.node(self.right_child_id).latex_string
        right_precedence_group = self.tree.node(self.right_child_id).precedence_group

        slot_precedence_group = raw_precedence_group+'_R'

        self.latex_string = self._replace_latex_string(latex_string, right_latex, slot_precedence_group, right_precedence_group, self.right_placeholder)

        if raw_precedence_group == "poly_prod" and (left_precedence_group == "poly_tens" or right_precedence_group == "poly_tens"):
            self.precedence_group = "poly_tens"
        else:
            self.precedence_group = raw_precedence_group

    def _custom_parenthesis_check(self, slot_precedence_group, in_precedence_group):
        left_rank = self.tree.node(self.left_child_id).out_rank
        if "poly_prod" in slot_precedence_group and "poly_tens" in in_precedence_group and left_rank>0:
            return True
        
        return False
    
    def generate_pseudocode_string(self):
        raw_pseudocode = self.grammar
        left_pseudocode = self.tree.node(self.left_child_id).pseudocode_string
        self.pseudocode_string = self._replace_pseudocode_string(raw_pseudocode, left_pseudocode, self.left_placeholder)
        right_pseudocode = self.tree.node(self.right_child_id).pseudocode_string
        self.pseudocode_string = self._replace_pseudocode_string(self.pseudocode_string, right_pseudocode, self.right_placeholder)


class PartialUnaryOperatorNode(TreeNode):

    left_placeholder = r"\ba\b"
    right_placeholder = r"\bi\b"
    compatible_children_groups = ("unary", "binary", "partial_unary", "poly")

    def __init__(self, node_id, tree, operator:list):
        super().__init__(node_id, tree)
        self.grammar = operator

        self.left_child_id = None
        self.left_child_rank = None
        self.right_child_id = None
        self.right_child_rank = None

        operator_info = self.tree.operators_dict_handler.get_info_from_operator(operator)
        try:
            self.output_rank_relation: str = operator_info["out_relation"]
            self.input_rank_possibilities: List[int] = operator_info["in_ranks"]
            self.output_rank_possibilities: List[int] = operator_info["out_ranks"]
        except:
            raise AttributeError(f"Incorrect typing in the entries of operator info list for {operator}")
    
    @classmethod
    def get_rank_table(cls, grammar_dict):
        rank_table=np.zeros((len(grammar_dict),5),dtype=int)
        for i, value in enumerate(grammar_dict.values()):
            rank_table[i,value["out_ranks"]]=1
        return rank_table

    @classmethod
    def create_new_node_as_child(cls, node_id, operator, parent_node, child_slot):
        new_nodes_list = []

        new_operator_node = cls(node_id, parent_node.tree, operator) # Instantiate the class
        new_operator_node.set_parent(parent_node, child_slot, node_id)
        new_operator_node.set_node_ranks()
        new_nodes_list.append(new_operator_node)

        node_id += 1
        new_coord_node = CoordNode(node_id, parent_node.tree)
        new_coord_node.set_parent(new_operator_node, 1, node_id)
        new_nodes_list.append(new_coord_node)

        return new_nodes_list

    @classmethod
    def create_new_node_as_root(cls, node_id, tree, operator, manual_rank):
        new_nodes_list = []

        new_operator_node = cls(node_id, tree, operator) # Instantiate the class
        new_operator_node.set_first_node_ranks(manual_rank)
        new_nodes_list.append(new_operator_node)

        node_id += 1
        new_coord_node = CoordNode(node_id, tree)
        new_coord_node.set_parent(new_operator_node, 1, node_id)
        new_nodes_list.append(new_coord_node)

        return new_nodes_list

    def get_child_rank(self, child_slot):
        if child_slot == 0:
            return self.left_child_rank
        elif child_slot == 1:
            return self.right_child_rank
        else:
            raise ValueError(f"Child slot {child_slot} does not apply to PartialUnaryOperatorNode")
    
    def set_child_id(self, node_id, child_slot):
        if child_slot == 0:
            self.left_child_id = node_id
        elif child_slot == 1:
            self.right_child_id = node_id
        else:
            raise ValueError(f"Child slot {child_slot} does not apply to PartialUnaryOperatorNode")

    def set_first_node_ranks(self, manual_rank):
        if manual_rank is None:
            self.out_rank = Utilities.get_random_element(self.output_rank_possibilities)[0]
        else:
            self.out_rank = manual_rank
        self.set_node_ranks()

    def set_node_ranks(self):
        self.right_child_rank = -1

        if self.out_rank is not None and self.left_child_rank is None:
            rank_calc_string = self.output_rank_relation+f"=={self.out_rank}"
            for trial_rank in self.input_rank_possibilities:
                if eval(re.sub(self.left_placeholder, str(trial_rank), rank_calc_string)):
                    self.left_child_rank = trial_rank
                    break
            assert self.left_child_rank is not None, f"No compatible child rank was found for operator {self.grammar} with out_rank {self.out_rank}"
        
        else:
            raise ValueError(f"Calling set_node_ranks with out_rank={self.out_rank} and left_child_rank={self.left_child_rank} is not possible for Unary Operator Node.")

    def has_free_spot(self):
        return self.left_child_id is None

    def get_child_candidates(self):
        child_slot = 0
        child_candidates = self.tree.operators_dict_handler.get_compatible_operators_by_rank(self.left_child_rank, self.compatible_children_groups)
        return child_candidates, child_slot
    
    def get_traversal_node_type(self):
        return "bifurcation"
        
    def create_variable_nodes(self, node_id):
        new_nodes_list = []
        if self.left_child_id is None:
            new_node = VariableNode(node_id, self.tree)
            new_node.set_parent(self, 0, node_id)
            new_node.get_var_id()
            new_nodes_list.append(new_node)
        return new_nodes_list
    
    def generate_latex_string(self):
        raw_latex, raw_precedence_group = self.tree.operators_dict_handler.get_operator_latex(self.grammar)

        # Left child:
        left_latex = self.tree.node(self.left_child_id).latex_string
        left_precedence_group = self.tree.node(self.left_child_id).precedence_group
        slot_precedence_group = raw_precedence_group+'_L'

        latex_string = self._replace_latex_string(raw_latex, left_latex, slot_precedence_group, left_precedence_group, self.left_placeholder)

        # Right child:
        right_latex = self.tree.node(self.right_child_id).latex_string
        right_precedence_group = self.tree.node(self.right_child_id).precedence_group

        slot_precedence_group = raw_precedence_group+'_R'

        self.latex_string = self._replace_latex_string(latex_string, right_latex, slot_precedence_group, right_precedence_group, self.right_placeholder)

        self.precedence_group = raw_precedence_group
    
    def generate_pseudocode_string(self):
        raw_pseudocode = self.grammar
        left_pseudocode = self.tree.node(self.left_child_id).pseudocode_string
        self.pseudocode_string = self._replace_pseudocode_string(raw_pseudocode, left_pseudocode, self.left_placeholder)
        right_pseudocode = self.tree.node(self.right_child_id).pseudocode_string
        self.pseudocode_string = self._replace_pseudocode_string(self.pseudocode_string, right_pseudocode, self.right_placeholder)
        
class PolyOperatorNode(TreeNode):

    left_placeholder = r"\ba\b"
    compatible_children_groups = ("unary", "binary", "partial_unary", "poly")

    def __init__(self, node_id, tree, operator:str, is_mono: bool = False):
        super().__init__(node_id, tree)
        self.grammar = operator

        self.left_child_id = None
        self.left_child_rank = None
        self.is_mono = is_mono

        operator_info = self.tree.operators_dict_handler.get_info_from_operator(operator)
        try:
            self.input_rank_possibilities: List[int] = operator_info["in_ranks"]
            self.output_rank_possibilities: List[int] = operator_info["out_ranks"]
        except:
            raise AttributeError(f"Incorrect typing in the entries of operator info list for {operator}")
    
    @classmethod
    def get_rank_table(cls, grammar_dict):
        rank_table=np.zeros((len(grammar_dict),5),dtype=int)
        for i, value in enumerate(grammar_dict.values()):
            rank_table[i,value["out_ranks"]]=1
        return rank_table
    
    @classmethod
    def create_new_node_as_child(cls, node_id, operator, parent_node, child_slot):

        new_nodes_list = []

        num_arguments = Utilities.get_random_element(list(range(1,5)))[0]
        is_mono = num_arguments == 1
        new_func_node = cls(node_id, parent_node.tree, operator, is_mono) # Instantiate the class
        new_func_node.set_parent(parent_node, child_slot, node_id)
        new_func_node.set_node_ranks()
        new_nodes_list.append(new_func_node)
        
        arggroup_operator = parent_node.tree.operators_dict_handler.get_unique_operator_from_class(ArggroupNode)
        if num_arguments==2:
            node_id += 1
            new_nodes_list.extend(ArggroupNode.create_new_node_as_child(node_id, arggroup_operator, new_nodes_list[-1], 0, True))
        elif num_arguments>2:
            node_id += 1
            new_nodes_list.extend(ArggroupNode.create_new_node_as_child(node_id, arggroup_operator, new_nodes_list[-1], 0, False))
            num_arguments -= 1
            while num_arguments > 2:
                node_id += 1
                new_nodes_list.extend(ArggroupNode.create_new_node_as_child(node_id, arggroup_operator, new_nodes_list[-1], 1, False))
                num_arguments -= 1
            node_id += 1
            new_nodes_list.extend(ArggroupNode.create_new_node_as_child(node_id, arggroup_operator, new_nodes_list[-1], 1, True))

        return new_nodes_list
    
    def get_child_rank(self, child_slot):
        assert child_slot == 0, f"Child slot {child_slot} does not apply to PolyOperatorNode"
        return self.left_child_rank
    
    def set_child_id(self, node_id, child_slot):
        assert child_slot == 0, f"Child slot {child_slot} does not apply to PolyOperatorNode"
        self.left_child_id = node_id

    def set_node_ranks(self):
        if self.out_rank is not None and self.left_child_rank is None:
            if self.is_mono:
                self.left_child_rank = Utilities.get_random_element(self.input_rank_possibilities)[0]
            else:
                self.left_child_rank = -1
                
        elif self.out_rank is None and self.left_child_rank is not None:
            self.out_rank = self.left_child_rank = Utilities.get_random_element(self.output_rank_possibilities)[0]
        else:
            raise ValueError(f"Calling set_node_ranks with out_rank={self.out_rank} and left_child_rank={self.left_child_rank} is not possible for Poly Operator Node.")

    def has_free_spot(self):
        return self.left_child_id is None

    def get_child_candidates(self):
        child_slot = 0
        child_candidates = self.tree.operators_dict_handler.get_compatible_operators_by_rank(self.left_child_rank, self.compatible_children_groups)
        return child_candidates, child_slot
    
    def get_traversal_node_type(self):
        return "pass"
        
    def create_variable_nodes(self, node_id):
        new_nodes_list = []
        if self.left_child_id is None:
            new_node = VariableNode(node_id, self.tree)
            new_node.set_parent(self, 0, node_id)
            new_node.get_var_id()
            new_nodes_list.append(new_node)
        return new_nodes_list
    
    def generate_latex_string(self):
        raw_latex, raw_precedence_group = self.tree.operators_dict_handler.get_operator_latex(self.grammar)
        left_latex = self.tree.node(self.left_child_id).latex_string
        left_precedence_group = self.tree.node(self.left_child_id).precedence_group
        
        slot_precedence_group = raw_precedence_group
        if slot_precedence_group[-2:] not in ('_L','_R'):
            slot_precedence_group = slot_precedence_group+'_L'

        self.latex_string = self._replace_latex_string(raw_latex, left_latex, slot_precedence_group, left_precedence_group, self.left_placeholder)
        
        self.precedence_group = raw_precedence_group
    
    def generate_pseudocode_string(self):
        raw_pseudocode = self.grammar
        left_pseudocode = self.tree.node(self.left_child_id).pseudocode_string
        self.pseudocode_string = self._replace_pseudocode_string(raw_pseudocode, left_pseudocode, self.left_placeholder)
        
    
class ArggroupNode(TreeNode):

    left_placeholder = r"\ba\b"
    right_placeholder = r"\bb\b"
    compatible_children_groups = ("unary", "binary", "partial_unary", "poly")

    def __init__(self, node_id, tree, operator:str, is_final: bool = False):
        super().__init__(node_id, tree)
        self.grammar = operator

        self.left_child_id = None
        self.left_child_rank = None
        self.right_child_id = None
        self.right_child_rank = None
        self.is_final = is_final

        operator_info = self.tree.operators_dict_handler.get_info_from_operator(operator)
        try:
            self.input_rank_possibilities: List[List[int]] = operator_info["in_ranks"]
        except:
            raise AttributeError(f"Incorrect typing in the entries of operator info list for {operator}")
    
    @classmethod
    def get_rank_table(cls, grammar_dict):
        rank_table=None
        return rank_table
        
    @classmethod
    def create_new_node_as_child(cls, node_id, operator, parent_node, child_slot, is_final):
        new_node = cls(node_id, parent_node.tree, operator, is_final) # Instantiate the class
        new_node.set_parent(parent_node, child_slot, node_id)
        new_node.set_node_ranks()
        return [new_node]

    def get_child_rank(self, child_slot):
        if child_slot == 0:
            return self.left_child_rank
        elif child_slot == 1:
            return self.right_child_rank
        else:
            raise ValueError(f"Child slot {child_slot} does not apply to BinaryOperatorNode")
    
    def set_child_id(self, node_id, child_slot):
        if child_slot == 0:
            self.left_child_id = node_id
        elif child_slot == 1:
            self.right_child_id = node_id
        else:
            raise ValueError(f"Child slot {child_slot} does not apply to BinaryOperatorNode")

    def set_node_ranks(self):
        if self.out_rank is not None and self.left_child_rank is None and self.right_child_rank is None:
            self.left_child_rank = Utilities.get_random_element(self.input_rank_possibilities[0])[0]
            if self.is_final:
                self.right_child_rank = Utilities.get_random_element(self.input_rank_possibilities[1])[0]
            else:
                self.right_child_rank = -1
        else:
            raise ValueError(f"Invalid combination of rank values when calling set_node_ranks in Arggroup: out_rank={self.out_rank}, left_child_rank={self.left_child_rank}, right_child_rank={self.right_child_rank}.")
    
    def has_free_spot(self):
        return self.left_child_id is None or self.right_child_id is None
    
    def get_child_candidates(self):

        if self.left_child_id is None and self.right_child_id is None:
            if Utilities.get_chance(0.5):
                child_slot = 0
                child_rank = self.left_child_rank
            else:
                child_slot = 1
                child_rank = self.right_child_rank
        elif self.left_child_id is None:
            child_slot = 0
            child_rank = self.left_child_rank
        elif self.right_child_id is None:
            child_slot = 1
            child_rank = self.right_child_rank
        else:
            raise ValueError("Either left or right child ID whould be None in order to get a child node in BinaryOperatorNode")
            
        child_candidates = self.tree.operators_dict_handler.get_compatible_operators_by_rank(child_rank, self.compatible_children_groups)
        
        return child_candidates, child_slot
    
    def get_traversal_node_type(self):
        return "bifurcation"
        
    def create_variable_nodes(self, node_id):
        new_nodes_list = []
        if self.left_child_id is None:
            new_node = VariableNode(node_id, self.tree)
            new_node.set_parent(self, 0, node_id)
            new_node.get_var_id()
            new_nodes_list.append(new_node)
            node_id+=1
        if self.right_child_id is None:
            new_node = VariableNode(node_id, self.tree)
            new_node.set_parent(self, 1, node_id)
            new_node.get_var_id()
            new_nodes_list.append(new_node)
        return new_nodes_list
    
    def generate_latex_string(self):
        raw_latex, raw_precedence_group = self.tree.operators_dict_handler.get_operator_latex(self.grammar)

        # Left child:
        left_latex = self.tree.node(self.left_child_id).latex_string
        left_precedence_group = self.tree.node(self.left_child_id).precedence_group
        slot_precedence_group = raw_precedence_group+'_L'

        latex_string = self._replace_latex_string(raw_latex, left_latex, slot_precedence_group, left_precedence_group, self.left_placeholder)

        # Right child:
        right_latex = self.tree.node(self.right_child_id).latex_string
        right_precedence_group = self.tree.node(self.right_child_id).precedence_group

        slot_precedence_group = raw_precedence_group+'_R'

        self.latex_string = self._replace_latex_string(latex_string, right_latex, slot_precedence_group, right_precedence_group, self.right_placeholder)

        self.precedence_group = raw_precedence_group

    def generate_pseudocode_string(self):
        raw_pseudocode = self.grammar
        left_pseudocode = self.tree.node(self.left_child_id).pseudocode_string
        self.pseudocode_string = self._replace_pseudocode_string(raw_pseudocode, left_pseudocode, self.left_placeholder)
        right_pseudocode = self.tree.node(self.right_child_id).pseudocode_string
        self.pseudocode_string = self._replace_pseudocode_string(self.pseudocode_string, right_pseudocode, self.right_placeholder)

class VariableNode(TreeNode):

    def __init__(self, node_id, tree):
        super().__init__(node_id, tree)
        self.grammar = "var"
        self.var_id = None
    
    def get_var_id(self):
        previous_vars = self.tree.variables_dict[self.out_rank]
        previous_vars_len = len(previous_vars.keys())
        if previous_vars_len==0:
            symbol = self.tree.variables_dict_handler.select_new_symbol_by_rank(self.out_rank)
            previous_vars[0] = {"node_ids": [self.node_id], "symbol": symbol}
            self.var_id = 0
        else:
            choice = Utilities.get_random_choice([0.5, *[0.5/previous_vars_len for i in range(previous_vars_len)]])
            if choice == 0:
                symbol = self.tree.variables_dict_handler.select_new_symbol_by_rank(self.out_rank)
                previous_vars[previous_vars_len] = {"node_ids": [self.node_id], "symbol": symbol}
                self.var_id = previous_vars_len
            else:
                previous_vars[choice-1]["node_ids"].append(self.node_id)
                self.var_id = choice-1

        self.grammar += str(self.out_rank)+"_"+str(self.var_id)


    def get_traversal_node_type(self):
        return "leaf"
    
    def has_free_spot(self):
        return False
    
    def create_variable_nodes(self, node_id):
        return []

    def generate_latex_string(self):
        self.latex_string = self.tree.variables_dict[self.out_rank][self.var_id]["symbol"][1]
        self.precedence_group = "mono"
    
    def generate_pseudocode_string(self):
        self.pseudocode_string = self.grammar

class CoordNode(TreeNode):

    def __init__(self, node_id, tree):
        super().__init__(node_id, tree)
        self.coord_id = Utilities.get_random_element([0,1,2])[0]
        self.grammar = "coord_"+str(self.coord_id)

    def get_traversal_node_type(self):
        return "leaf"
    
    def has_free_spot(self):
        return False
    
    def create_variable_nodes(self, node_id):
        return []
    
    def generate_latex_string(self):
        self.latex_string = self.tree.coords_dict_handler.get_latex_coord(self.coord_id)
        self.precedence_group = "mono"

    def generate_pseudocode_string(self):
        self.pseudocode_string = self.grammar

