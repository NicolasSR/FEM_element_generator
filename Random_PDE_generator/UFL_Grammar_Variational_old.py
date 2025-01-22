import numpy as np
import re
from copy import deepcopy

from bigtree import list_to_tree, tree_to_dot

from seeded_random_generator import SeededRandomGenerator as Utilities
from Random_PDE_generator.latex_parser_old import LatexParser, conversions_list, parenthesis_table, categories_map

from datasets import Dataset

# This is a dataset containing the specific notation to be used by our LLM in order to generate pseudocode for PDEs
# It includes the operator notation, the input rank limitations, the output rank (as afunction of the input), and a description
# We are targeting only variational forms of PDEs in the context of FEM. Therefore, the dimensions will be fixed by the domain
#   so we don't need to care about them

grammar_list=[
# Scalar specific
['pow(a,b)', 'a==b==0', 'a', [0,0], [0], "a^b"],
['sqrt(a)', 'a==0', 'a', [0], [0], "square root of a"],
['exp(a)', 'a==0', 'a', [0], [0], "exponential of scalar"],
['ln(a)', 'a==0', 'a', [0], [0], "natural log of scalar"],
['abs(a)', 'a==0', 'a', [0], [0], "absolute value of scalar"],
['sign(a)', 'a==0', 'a', [0], [0], "sign of scalar (1 or -1)"],
    # Maybe add trigonometric operators

# Semi-Scalar specific
['prod(a,b)', 'min(a,b)==0', 'max(a,b)', [[0,1,2,3,4],[0,1,2,3,4]], [0,1,2,3,4], "ab, aB, Ab..."],
['frac(a,b)', 'b==0', 'a', [[0,1,2,3,4],0], [0,1,2,3,4], "a/b"],

# Vector specific
['dot(a,b)', 'a==b==1', 'a-b', [1,1], [0], "dot product of 2 vectors"],
['cross(a,b)', 'a==b==1', 'a', [1,1], [1], "cross product of 2 vectors"],

# Matrix specific
['matrix_prod(a,b)', 'a==b==2', 'a', [2,2], [2], "classic matrix product: AB_ik = A_ij*B_jk"],
['transpose(a)', 'a==2', 'a', [2], [2], "a^T"],
['sym(a)', 'a==2', 'a', [2], [2], "a=1/2(a+a^T)"],
['skew(a)', 'a==2', 'a', [2], [2], "a=1/2(a-a^T)"],
['dev(a)', 'a==2', 'a', [2], [2], "deviatoric part of a (check Cauchy Stress)"],
['tr(a)', 'a==2', '0', [2], [0], "trace of matrix"],
['det(a)', 'a==2', '0', [2], [0], "determinant of matrix"], # Maybe add or substitute by general tensor hyperdeterminant

# General
['plus(a,b)', 'a==b', 'a', [[0,1,2,3,4],-1], [0,1,2,3,4], "a+b"],  # -1 implies that b takes the same rank as a
['outer(a,b)', 'min(a,b)>0', 'a+b', [[1,2],[1,2]], [2,3,4], "tensor product: c_ij=a_i*b_j; c_ijk=a_ij*b_k"],
['contract(a,b)', 'min(a,b)>0', 'max(a,b)-min(a,b)', [[1,2,3,4],[1,2,3,4]], [0,1,2,3], "cotract tensors: c_i=a_ijk*b_jk, c_k=a_ij*b_ijk"],
['inner(a,b)', 'a==b>0', 'a-b', [[1,2,3,4],-1], [0], "<a,b> = a_ij*b_ij"], # Maybe redundant compared to contraction

# Differential operators, i represents the component with respect to which we differenciate
['partial(a,i)', 'a>=0', 'a', [[0,1,2,3,4]], [0,1,2,3,4], 'Partial derivative over one component of the domain space'],
['der_n(a)', 'a==1', 'a-1', [1], [0], 'Dot product of gradient of vector a with normal vector'],
['grad(a)', 'a==0', 'a+1', [0], [1], 'gradient of a'], # Maybe we should generalize to more than scalars
['div(a)', 'a==1', 'a-1', [1], [0], 'divergence of a'], # Maybe we should generalize to more than scalars
['curl(a)', 'a==1', 'a', [1], [1], 'divergence of a'], # Maybe we should generalize to more than scalars

# User-defined functions
['fun(a)', '0==0', '0==0', [[0,1,2,3,4]], [0,1,2,3,4], 'Generic, user-defined function'],
['a,b', '0', '0', [[0,1,2,3,4],[0,1,2,3,4]], [-1], 'Argument group. To be able to add more arguments into user-defined functions. Cannot be selected randomly']
]

# Now we define a set of possible symbols for our variables:
alphabet_with_greek = {
    # Rank: [[pseudocode,latex],...]
    # '0': [["a", "a"], ["b", "b"], ["c", "c"], ["d", "d"], ["e", "e"], ["f", "f"], ["g", "g"],
    '0': [["d", "d"], ["e", "e"], ["f", "f"], ["g", "g"],
    ["h", "h"], ["i", "i"], ["j", "j"], ["k", "k"], ["l", "l"], ["m", "m"], ["n", "n"],
    ["o", "o"], ["p", "p"], ["q", "q"], ["r", "r"], ["s", "s"], ["t", "t"], ["u", "u"],
    ["v", "v"], ["w", "w"], ["x", "x"], ["y", "y"], ["z", "z"],
    ["alpha", ":/:alpha"], ["beta", ":/:beta"], ["gamma", ":/:gamma"],
    ["delta", ":/:delta"], ["epsilon", ":/:epsilon"], ["zeta", ":/:zeta"],
    ["eta", ":/:eta"], ["theta", ":/:theta"], ["kappa", ":/:kappa"],
    ["lambda", ":/:lambda"], ["mu", ":/:mu"], ["nu", ":/:nu"], ["xi", ":/:xi"],
    ["pi", ":/:pi"], ["rho", ":/:rho"], ["sigma", ":/:sigma"], ["tau", ":/:tau"],
    ["upsilon", ":/:upsilon"], ["phi", ":/:phi"], ["chi", ":/:chi"],
    ["psi", ":/:psi"], ["omega", ":/:omega"], ["varphi", ":/:varphi"],
    ["varepsilon", ":/:varepsilon"]],
    # '1': [["a_bold", ":/:mathbf{a}"], ["b_bold", ":/:mathbf{b}"], ["c_bold", ":/:mathbf{c}"], ["d_bold", ":/:mathbf{d}"], ["e_bold", ":/:mathbf{e}"], ["f_bold", ":/:mathbf{f}"], ["g_bold", ":/:mathbf{g}"],
    '1': [["c_bold", ":/:mathbf{c}"], ["d_bold", ":/:mathbf{d}"], ["e_bold", ":/:mathbf{e}"], ["f_bold", ":/:mathbf{f}"], ["g_bold", ":/:mathbf{g}"],
    ["h_bold", ":/:mathbf{h}"], ["i_bold", ":/:mathbf{i}"], ["j_bold", ":/:mathbf{j}"], ["k_bold", ":/:mathbf{k}"], ["l_bold", ":/:mathbf{l}"], ["m_bold", ":/:mathbf{m}"], ["n_bold", ":/:mathbf{n}"],
    ["o_bold", ":/:mathbf{o}"], ["p_bold", ":/:mathbf{p}"], ["q_bold", ":/:mathbf{q}"], ["r_bold", ":/:mathbf{r}"], ["s_bold", ":/:mathbf{s}"], ["t_bold", ":/:mathbf{t}"], ["u_bold", ":/:mathbf{u}"],
    ["v_bold", ":/:mathbf{v}"], ["w_bold", ":/:mathbf{w}"], ["x_bold", ":/:mathbf{x}"], ["y_bold", ":/:mathbf{y}"], ["z_bold", ":/:mathbf{z}"],
    ["alpha_bold", ":/:boldsymbol{:/:alpha}"], ["beta_bold", ":/:boldsymbol{:/:beta}"], ["gamma_bold", ":/:boldsymbol{:/:gamma}"],
    ["delta_bold", ":/:boldsymbol{:/:delta}"], ["epsilon_bold", ":/:boldsymbol{:/:epsilon}"], ["zeta_bold", ":/:boldsymbol{:/:zeta}"],
    ["eta_bold", ":/:boldsymbol{:/:eta}"], ["theta_bold", ":/:boldsymbol{:/:theta}"], ["kappa_bold", ":/:boldsymbol{:/:kappa}"],
    ["lambda_bold", ":/:boldsymbol{:/:lambda}"], ["mu_bold", ":/:boldsymbol{:/:mu}"], ["nu_bold", ":/:boldsymbol{:/:nu}"], ["xi_bold", ":/:boldsymbol{:/:xi}"],
    ["pi_bold", ":/:boldsymbol{:/:pi}"], ["rho_bold", ":/:boldsymbol{:/:rho}"], ["sigma_bold", ":/:boldsymbol{:/:sigma}"], ["tau_bold", ":/:boldsymbol{:/:tau}"],
    ["upsilon_bold", ":/:boldsymbol{:/:upsilon}"], ["phi_bold", ":/:boldsymbol{:/:phi}"], ["chi_bold", ":/:boldsymbol{:/:chi}"],
    ["psi_bold", ":/:boldsymbol{:/:psi}"], ["omega_bold", ":/:boldsymbol{:/:omega}"], ["varphi_bold", ":/:boldsymbol{:/:varphi}"],
    ["varepsilon_bold", ":/:boldsymbol{:/:varepsilon}"]],
    '2,3,4': [["A_bold", ":/:mathbf{A}"], ["B_bold", ":/:mathbf{B}"], ["C_bold", ":/:mathbf{C}"], ["D_bold", ":/:mathbf{D}"], ["E_bold", ":/:mathbf{E}"], ["F_bold", ":/:mathbf{F}"], ["G_bold", ":/:mathbf{G}"],
    ["H_bold", ":/:mathbf{H}"], ["I_bold", ":/:mathbf{I}"], ["J_bold", ":/:mathbf{J}"], ["K_bold", ":/:mathbf{K}"], ["L_bold", ":/:mathbf{L}"], ["M_bold", ":/:mathbf{M}"], ["N_bold", ":/:mathbf{N}"],
    ["O_bold", ":/:mathbf{O}"], ["P_bold", ":/:mathbf{P}"], ["Q_bold", ":/:mathbf{Q}"], ["R_bold", ":/:mathbf{R}"], ["S_bold", ":/:mathbf{S}"], ["T_bold", ":/:mathbf{T}"], ["U_bold", ":/:mathbf{U}"],
    ["V_bold", ":/:mathbf{V}"], ["W_bold", ":/:mathbf{W}"], ["X_bold", ":/:mathbf{X}"], ["Y_bold", ":/:mathbf{Y}"], ["Z_bold", ":/:mathbf{Z}"],
    ["Gamma_bold", ":/:boldsymbol{:/:Gamma}"], ["Delta_bold", ":/:boldsymbol{:/:Delta}"],
    ["Theta_bold", ":/:boldsymbol{:/:Theta}"], ["Lambda_bold", ":/:boldsymbol{:/:Lambda}"], ["Xi_bold", ":/:boldsymbol{:/:Xi}"],
    ["Pi_bold", ":/:boldsymbol{:/:Pi}"], ["Sigma_bold", ":/:boldsymbol{:/:Sigma}"], ["Upsilon_bold", ":/:boldsymbol{:/:Upsilon}"],
    ["Phi_bold", ":/:boldsymbol{:/:Phi}"], ["Psi_bold", ":/:boldsymbol{:/:Psi}"], ["Omega_bold", ":/:boldsymbol{:/:Omega}"]]
}

class OperatorsTree():

    def __init__(self, operators_list, first_operator_id, manual_rank=None):
        # print('TREE INITIALIZED')
        self.operators_list = operators_list
        self.rank_table = self._generate_rank_table()
        self.nodes_list = [TreeNode(0, self.operators_list[first_operator_id])]
        self.node(0).set_first_node_ranks(manual_rank)
        self.root = 0
        self.candidate_children = {0}
        self.leaf_nodes_list = None
        self.upwards_traversal_order = None

        self.fun_id, self.arggroup_id = self._get_user_function_operator_id()
    
    def _get_user_function_operator_id(self):
        fun_id = None
        arggroup_id = None
        for id, operator in enumerate(self.operators_list):
            if operator[0] == 'fun(a)':
                fun_id = id
            elif operator[0] == 'a,b':
                arggroup_id = id
        return fun_id, arggroup_id

    def add_new_child_node(self, operator_id, parent_id, new_node_rank, child_slot):
        node_id=len(self.nodes_list)
        new_node = TreeNode(node_id, self.operators_list[operator_id])
        new_node.set_parent(parent_id)
        compute_second_child_rank = len(new_node.children_ids)>1
        new_node.get_children_ranks(new_node_rank, compute_second_child_rank)
        if self.nodes_list[parent_id].add_child_id_in_slot(new_node.node_id, child_slot):
            self.candidate_children.remove(parent_id)
        self.candidate_children.add(new_node.node_id)
        self.nodes_list.append(new_node)

    def add_new_child_node_as_user_function(self, parent_id, new_node_rank, child_slot):
        num_arguments = Utilities.get_random_element(list(range(1,5)))[0]
        # Add func node first
        node_id=len(self.nodes_list)
        new_node = TreeNode(node_id, self.operators_list[self.fun_id])
        new_node.set_parent(parent_id)
        if self.node(parent_id).add_child_id_in_slot(node_id, child_slot=child_slot):
            self.candidate_children.remove(parent_id)
        if num_arguments == 1:
            new_node.get_children_ranks_func(new_node_rank, continue_groups_chain = False)
            self.candidate_children.add(node_id)
            self.nodes_list.append(new_node)
        else:
            new_node.get_children_ranks_func(new_node_rank, continue_groups_chain = True)
            self.nodes_list.append(new_node)
            while num_arguments > 1:
                # If even more arguments are required, add arggroups one after another (always on the right slot)
                parent_id = node_id
                node_id=len(self.nodes_list)
                new_node = TreeNode(node_id, self.operators_list[self.arggroup_id])
                new_node.set_parent(parent_id)
                children_ids = self.node(parent_id).children_ids
                self.node(parent_id).add_child_id_in_slot(node_id, child_slot=min(len(children_ids)-1,1))
                self.candidate_children.add(node_id)
                new_node.get_children_ranks_arggroup(continue_groups_chain=num_arguments>2)
                self.nodes_list.append(new_node)
                num_arguments-=1

    def add_new_root_node(self, operator_id, child_node_id):
        node_id=len(self.nodes_list)
        new_node = TreeNode(node_id, self.operators_list[operator_id])
        self.nodes_list[self.root].set_parent(node_id)
        if new_node.add_child_id_in_slot(child_node_id):
            new_node.get_own_rank(self.nodes_list[self.root].rank, compute_second_child_rank=False)
        else:
            self.candidate_children.add(new_node.node_id)
            new_node.get_own_rank(self.nodes_list[self.root].rank, compute_second_child_rank=True)
        self.root = node_id
        self.nodes_list.append(new_node)

    def add_new_root_node_as_user_function(self, child_node_id):
        num_arguments = Utilities.get_random_element(list(range(1,5)))[0]
        if num_arguments>1:
            # Add new arggroup node as parent
            node_id=len(self.nodes_list)
            self.nodes_list[self.root].set_parent(node_id)
            left_rank=self.nodes_list[self.root].rank
            self.root = node_id
            new_node = TreeNode(node_id, self.operators_list[self.arggroup_id])
            new_node.add_child_id_in_slot(child_node_id, child_slot=0)
            new_node.get_own_rank_arggroup(left_rank, continue_groups_chain=num_arguments>2)
            self.nodes_list.append(new_node)
            num_arguments-=2
            while num_arguments > 0:
                # If even more arguments are required, add arggroups one after another (always on the right slot)
                parent_id = node_id
                node_id=len(self.nodes_list)
                new_node = TreeNode(node_id, self.operators_list[self.arggroup_id])
                new_node.set_parent(parent_id)
                self.node(parent_id).add_child_id_in_slot(node_id, child_slot=1)
                self.candidate_children.add(node_id)
                new_node.get_children_ranks_arggroup(continue_groups_chain=num_arguments>1)
                self.nodes_list.append(new_node)
                num_arguments-=1

        # Add func node on top of everything
        node_id=len(self.nodes_list)
        self.node(self.root).set_parent(node_id)
        child_rank=self.node(self.root).rank
        new_node = TreeNode(node_id, self.operators_list[self.fun_id])
        new_node.add_child_id_in_slot(self.root, child_slot=0)
        self.root = node_id
        new_node.get_own_rank_func(child_rank)
        self.nodes_list.append(new_node)

    def get_random_new_node(self, update_root_too):
        candidate_nodes = list(self.candidate_children)
        if update_root_too:
            candidate_nodes.append(-1)
        chosen_associated_node = Utilities.get_random_element(candidate_nodes)[0]
        # print('\nCandidate nodes: ', candidate_nodes, ' Chosen node: ', chosen_associated_node)
        if chosen_associated_node == -1:
            # print('Generating new root node')
            first_child_compat_mask = self.rank_table[:,self.node(self.root).rank,0]==1
            candidate_operator_ids = np.arange(len(self.operators_list))[first_child_compat_mask]
            operator_id = Utilities.get_random_element(candidate_operator_ids)[0]
            if operator_id == self.fun_id:
                self.add_new_root_node_as_user_function(operator_id, chosen_associated_node)
            else:
                self.add_new_root_node(operator_id, chosen_associated_node)
        else:
            associated_node_children = self.node(chosen_associated_node).children_ids
            child_slot = Utilities.get_random_element([x for x,y in associated_node_children.items() if y is None])[0]
            new_node_rank = self.node(chosen_associated_node).children_ranks[child_slot]
            # print(new_node_rank)
            root_compat_mask = self.rank_table[:,new_node_rank,2]==1
            candidate_operator_ids = np.arange(len(self.operators_list))[root_compat_mask]
            operator_id = Utilities.get_random_element(candidate_operator_ids)[0]
            # print('Chosen_slot: ', child_slot)
            if operator_id == self.fun_id:
                self.add_new_child_node_as_user_function(chosen_associated_node, new_node_rank, child_slot)
            else:
                self.add_new_child_node(operator_id, chosen_associated_node, new_node_rank, child_slot)

    def node(self,i):
        return self.nodes_list[i]

    def _generate_rank_table(self):
        rank_table=np.zeros((len(self.operators_list),5,3),dtype=int)
        for i, operator in enumerate(self.operators_list):
            rank_table[i,operator[3][0],0]=1
            if operator[4]!=[-1]:
                rank_table[i,operator[4],2]=1
            if len(operator[3]) == 2:
                if operator[3][1] == -1:
                    rank_table[i,operator[3][0],1]=1
                else:
                    rank_table[i,operator[3][1],1]=1
        return rank_table
    
    def get_variables_with_rank(self):
        all_variables={0:[],1:[],2:[],3:[],4:[]}
        for node_id in self.candidate_children:
            node = self.node(node_id)
            for slot, value in node.children_ids.items():
                if value is None:
                    all_variables[node.children_ranks[slot]].append([node_id,slot])
        return all_variables

    def visualize_tree(self):
        
        all_paths=[]
        child_slot_map = {0.1: 0, 0.2: 1}
        
        # print(self.upwards_traversal_order)
        for raw_id in self.upwards_traversal_order:
            variable_indicator = round(raw_id % 1,1)
            if  variable_indicator != 0:
                node_id = int(round(raw_id-variable_indicator))
                current_node = self.node(node_id)
                variable_rank = current_node.children_ranks[child_slot_map[variable_indicator]]
                if node_id == self.root:
                    all_paths.append(str(node_id)+' ['+str(current_node.rank)+','+current_node.operator_info[0]+']/'+str(variable_rank+variable_indicator))
                    continue
                current_parent_id = current_node.parent_id
                current_parent = self.node(current_parent_id)
                current_str = str(current_parent_id)+' ['+str(current_parent.rank)+','+current_parent.operator_info[0]+']/'+str(node_id)+' ['+str(current_node.rank)+','+current_node.operator_info[0]+']/'+str(variable_rank+variable_indicator)
                while current_parent_id != self.root:
                    current_parent_id = self.node(current_parent_id).parent_id
                    current_parent = self.node(current_parent_id)
                    current_str = str(current_parent_id)+' ['+str(current_parent.rank)+','+current_parent.operator_info[0]+']/'+current_str
                all_paths.append(current_str)

        # print(all_paths)
        graph = tree_to_dot(list_to_tree(all_paths), node_colour="white")
        graph.write_png("tree.png")

    def get_upwards_traversal_order_with_variables(self):
        
        self.leaf_nodes_list = []

        nodes_id_set = set([node.node_id for node in self.nodes_list])
        bifurcations = []
        stack=[]
        output_list = []

        current_node = self.node(self.root)

        while len(nodes_id_set) > 0:

            leaf=False
            while not leaf:
                nones_list = []
                for val in current_node.children_ids.values():
                    if val is None:
                        nones_list.append(0)
                    else:
                        nones_list.append(1)
                nones_list = np.array(nones_list)
                nones_list_indicator = np.sum(nones_list) + 0.1*len(nones_list)

                stack.append(current_node.node_id)
                nodes_id_set.remove(current_node.node_id)
                if nones_list_indicator == 0.1: # Leaf with single variable
                    leaf = True
                    self.leaf_nodes_list.append(current_node.node_id)
                    stack.append(current_node.node_id + 0.1)
                    break
                elif nones_list_indicator == 0.2: # Leaf with two variables
                    leaf = True
                    self.leaf_nodes_list.append(current_node.node_id)
                    stack.append(current_node.node_id + 0.2)
                    stack.append(current_node.node_id + 0.1)
                    break
                elif nones_list_indicator == 2.2:  # Bifurcation
                    bifurcations.append(current_node.node_id)
                elif nones_list_indicator==1.2 and nones_list[0]==0: # Left child is a symbol
                    output_list.append(current_node.node_id + 0.1)
                elif nones_list_indicator==1.2 and nones_list[0]==1: # Right child is a symbol
                    stack.append(current_node.node_id + 0.2)

                child_slot = np.where(nones_list==1)[0][0]
                next_node_id = current_node.children_ids[child_slot]

                current_node = self.node(next_node_id)

            bifurcation_found = False
            while len(stack)>0 and bifurcation_found == False:
                if stack[-1] in bifurcations:
                    bifurcation_found = True
                    next_node_id = self.node(stack[-1]).children_ids[1]
                    bifurcations.pop()
                else:
                    output_list.append(stack.pop())
            

            current_node = self.node(next_node_id)

        self.upwards_traversal_order = output_list
        
        # print('Left-up node order: ', self.upwards_traversal_order)
        # print('Left-to-right leaf nodes: ', self.leaf_nodes_list)


class TreeNode():
    
    def __init__(self, node_id: int, operator):
        self.node_id = node_id
        self.operator_info = operator
        self.parent_id = None
        self._get_empty_children_ids()
        self.latex_string = None

        # print(self.operator_info)
        # print('Parent id: ', self.parent_id, '. Children ids: ', self.children_ids)


    def set_parent(self, parent_id):
        self.parent_id = parent_id
    
    def add_child_id_in_slot(self, child_id, child_slot = 0):
        self.children_ids[child_slot]=child_id
        return not None in self.children_ids.values()

    def _get_empty_children_ids(self):
        self.children_ids = {}
        for i in range(len(self.operator_info[3])):
            self.children_ids[i] = None

    def set_first_node_ranks(self, manual_rank):
        if manual_rank is None:
            output_rank = Utilities.get_random_element(self.operator_info[4])[0]
        else:
            output_rank = manual_rank
        compute_second_child_rank = len(self.operator_info[3])==2
        self.get_children_ranks(output_rank, compute_second_child_rank)

    def get_own_rank(self, known_child_rank, compute_second_child_rank=False):
        self.children_ranks = [known_child_rank]
        if compute_second_child_rank:
            if type(self.operator_info[3][1]) == list:
                second_rank = Utilities.get_random_element(self.operator_info[3][1])[0]
            elif self.operator_info[3][1]==-1:
                second_rank = known_child_rank
            else:
                second_rank = self.operator_info[3][1]
            self.children_ranks.append(second_rank)
        
        argument_placeholders=[r"\ba\b",r"\bb\b"]
        rank_calc_string = self.operator_info[2]
        for i in range(len(self.children_ranks)):
            rank_calc_string = re.sub(argument_placeholders[i], str(self.children_ranks[i]), rank_calc_string)
        self.rank = eval(rank_calc_string)
        # print('Set node ranks to: ', self.rank, '; ', self.children_ranks)

    def get_own_rank_arggroup(self, known_child_rank, continue_groups_chain=False):
        self.children_ranks = [known_child_rank]
        if continue_groups_chain:
            self.children_ranks.append(-1)
        else:
            second_rank = Utilities.get_random_element(self.operator_info[3][1])[0]
            self.children_ranks.append(second_rank)
        self.rank = -1
        # print('Set node ranks to: ', self.rank, '; ', self.children_ranks)

    def get_own_rank_func(self, known_child_rank):
        self.children_ranks = [known_child_rank]
        self.rank = Utilities.get_random_element(self.operator_info[4])[0]
        # print('Set node ranks to: ', self.rank, '; ', self.children_ranks)

    def get_children_ranks(self, known_own_rank, compute_second_child_rank=False):
        self.rank = known_own_rank

        argument_placeholders=[r"\ba\b",r"\bb\b",r"\bc\b"]

        if not compute_second_child_rank:
            if type(self.operator_info[3][0]) == list:
                rank_calc_string = self.operator_info[2]+"==c"
                for trial_rank in self.operator_info[3][0]:
                    rank_calc_string_sub = re.sub(argument_placeholders[0], str(trial_rank), rank_calc_string)
                    if eval(re.sub(argument_placeholders[2], str(self.rank), rank_calc_string_sub)):
                        child_rank = trial_rank
            else:
                child_rank = self.operator_info[3][0]
            self.children_ranks=[child_rank]
        else:
            if type(self.operator_info[3][0]) == list:
                if type(self.operator_info[3][1]) == list:
                    combinations_grid = np.meshgrid(self.operator_info[3][0], self.operator_info[3][1])
                    combinations = np.hstack([combinations_grid[0].reshape((-1,1)),combinations_grid[1].reshape((-1,1))])
                elif self.operator_info[3][1] == -1:
                    combinations = np.array(self.operator_info[3][0]).reshape((-1,1))
                    combinations = np.hstack([combinations,combinations])
                else:
                    combinations = np.array(self.operator_info[3][0]).reshape((-1,1))
                    combinations = np.hstack([combinations,np.ones_like(combinations)*self.operator_info[3][1]])
                valid_combinations = []

                rank_calc_string_output = self.operator_info[2]+"==c"
                rank_calc_string_inputs = self.operator_info[1]
                for combo in combinations:
                    rank_calc_string_output_sub = re.sub(argument_placeholders[0], str(combo[0]), rank_calc_string_output)
                    rank_calc_string_output_sub = re.sub(argument_placeholders[1], str(combo[1]), rank_calc_string_output_sub)
                    rank_calc_string_output_sub = re.sub(argument_placeholders[2], str(self.rank), rank_calc_string_output_sub)
                    rank_calc_string_inputs_sub = re.sub(argument_placeholders[0], str(combo[0]), rank_calc_string_inputs)
                    rank_calc_string_inputs_sub = re.sub(argument_placeholders[1], str(combo[1]), rank_calc_string_inputs_sub)
                    if eval(rank_calc_string_output_sub) and eval(rank_calc_string_inputs_sub):
                        valid_combinations.append(combo)
                self.children_ranks = list(Utilities.get_random_element(valid_combinations)[0])
            else:
                self.children_ranks = [self.operator_info[3][0], self.operator_info[3][1]]
        
        # print('Set node ranks to: ', self.rank, '; ', self.children_ranks)

    def get_children_ranks_arggroup(self, continue_groups_chain=False):
        self.rank = -1
        self.children_ranks = [Utilities.get_random_element(self.operator_info[3][1])[0]]
        if continue_groups_chain:
            self.children_ranks.append(-1)
        else:
            second_rank = Utilities.get_random_element(self.operator_info[3][1])[0]
            self.children_ranks.append(second_rank)
        # print('Set node ranks to: ', self.rank, '; ', self.children_ranks)

    def get_children_ranks_func(self, known_own_rank, continue_groups_chain = False):
        self.rank = known_own_rank
        if continue_groups_chain:
            self.children_ranks = [-1]
        else:
            self.children_ranks = [Utilities.get_random_element(self.operator_info[3][0])[0]]
        # print('Set node ranks to: ', self.rank, '; ', self.children_ranks)

# Let's define a class that will generate a random sequence of these operators.
class RandomEquationGenerator():

    def __init__(self, grammar_list, symbols_list):
        self.grammar_list = grammar_list
        self.symbols_list = symbols_list

    def generate_sequence(self, iterations_range, update_root_too=False):

        root_operator_names = ['prod(a,b)', 'dot(a,b)', 'inner(a,b)']
        root_operator_ids = []
        for i, operator_info in enumerate(self.grammar_list):
            if operator_info[0] in root_operator_names:
                root_operator_ids.append(i)
        root_operator_id = Utilities.get_random_element(root_operator_ids)[0]
        # print(root_operator_id)

        manual_rank = 0
        tree = OperatorsTree(self.grammar_list, root_operator_id, manual_rank = manual_rank)
        
        iterations = Utilities.get_random_element(list(range(iterations_range[0],iterations_range[1]+1)))[0]
        for iter in range(iterations):
            tree.get_random_new_node(update_root_too)
        
        tree.get_upwards_traversal_order_with_variables()

        tree.visualize_tree()

        return tree

    def assign_variables(self, tree):
        all_variables = tree.get_variables_with_rank()
        # Utilities.set_seed(None)

        symbol_candidates = deepcopy(self.symbols_list)

        for var_rank, vars_list in all_variables.items():
            current_symbol_candidates=None
            for key in symbol_candidates.keys():
                if str(var_rank) in key:
                    current_symbol_candidates=symbol_candidates[key]
                    break
            total_elements = len(vars_list)
            if total_elements == 0:
                groups = None
            elif total_elements == 1:
                # groups = [0]
                symbol, symbol_id = Utilities.get_random_element(current_symbol_candidates)
                vars_list[0].append(symbol)
                current_symbol_candidates.pop(symbol_id)
            else:
                elements=list(range(total_elements))
                elements=Utilities.shuffle_elements(elements)
                
                # Generate random sizes that sum to total_elements
                sizes = []
                remaining = total_elements
                while remaining > 0:
                    size = Utilities.get_random_element(list(range(1, remaining+1)))[0]  # Random size between 1 and remaining
                    sizes.append(size)
                    remaining -= size
                
                # Create groups based on the random sizes
                # groups = []
                start_index = 0
                for size in sizes:
                    # groups.append(elements[start_index:start_index + size])
                    symbol, symbol_id = Utilities.get_random_element(current_symbol_candidates)
                    current_symbol_candidates.pop(symbol_id)
                    for element in elements[start_index:start_index + size]:
                        vars_list[element].append(symbol)
                    start_index += size
        
        # print(all_variables)

        variables_symbols = {}
        for equivalences_list in all_variables.values():
            for equiv in equivalences_list:
                variables_symbols[str([equiv[0],equiv[1]])]=equiv[2]
        # variables_symbols = {'[2,0]': 'eps'}
        return variables_symbols

    def convert_to_latex(self, tree, variables_symbols, conversions_list, parenthesis_table, categories_map):
        local_placeholders=[r"\ba\b",r"\bb\b"]
        notation_type_suffix={0:'_L',1:'_R'}

        conversion_dict={}
        for id, operator_info in enumerate(self.grammar_list):
            conversion_dict[operator_info[0]] = conversions_list[id]

        for raw_id in tree.upwards_traversal_order:
            if raw_id%1==0:
                node_id = raw_id
                current_node = tree.node(node_id)
                current_string = conversion_dict[current_node.operator_info[0]][0]
                for slot, child_id in current_node.children_ids.items():
                    if child_id is None:
                        current_symbol = variables_symbols[str([node_id,slot])][1]
                        current_string = re.sub(local_placeholders[slot], current_symbol, current_string)
                    else:
                        current_notation_type = conversion_dict[current_node.operator_info[0]][1]
                        print(current_notation_type)
                        exit()
                        if '_R' not in current_notation_type and '_L' not in current_notation_type:
                            current_notation_type+=notation_type_suffix[slot]
                        incoming_notation_type = conversion_dict[tree.node(child_id).operator_info[0]][1]
                        repl = tree.node(child_id).latex_string
                        needs_parenthesis = parenthesis_table[categories_map[0][incoming_notation_type],categories_map[1][current_notation_type]]
                        if  needs_parenthesis == 1 or (Utilities.get_chance(0.1) and current_node.operator_info[0]!=tree.operators_list[tree.arggroup_id][0]):
                            parenthesis_type = Utilities.get_random_choice([0.4,0.4,0.1,0.1])
                            if parenthesis_type==0:
                                repl = '('+repl+')'
                            elif parenthesis_type==1:
                                repl = ':/:left ('+repl+':/:right )'
                            elif parenthesis_type==2:
                                repl = '['+repl+']'
                            elif parenthesis_type==3:
                                repl = ':/:left ['+repl+':/:right ]'
                            else: print('Error, incorrect group choice for parenthesis format')
                        current_string = re.sub(local_placeholders[slot], repl, current_string)
                current_node.latex_string=current_string
            

        full_latex_string = tree.node(tree.root).latex_string
        full_latex_string = re.sub(':/:', r'\\\\', full_latex_string)
        return full_latex_string



    def convert_to_string(self, tree, variables_symbols, conversions_list = None):
        local_placeholders=[r"\ba\b",r"\bb\b"]

        conversion_dict={}
        if conversions_list is None:
            for operator_info in self.grammar_list:
                conversion_dict[operator_info[0]] = operator_info[0]
        else:
            for id, operator_info in enumerate(self.grammar_list):
                conversion_dict[operator_info[0]] = conversions_list[id]

        ids_to_replace = []

        def get_local_string(current_id):
            current_node = tree.node(current_id)
            current_string = conversion_dict[current_node.operator_info[0]]
            for slot, child_id in current_node.children_ids.items():
                if child_id is not None:
                    current_string = re.sub(local_placeholders[slot], str(child_id), current_string)
                    ids_to_replace.append(child_id)
                else:
                    current_string = re.sub(local_placeholders[slot], variables_symbols[str([current_id,slot])][0], current_string)
            if conversions_list is not None:
                current_string = '('+current_string+')'
            return current_string
        
        full_string = get_local_string(tree.root)
        while len(ids_to_replace) > 0:
            current_id = ids_to_replace[0]
            local_string = get_local_string(current_id)
            full_string = re.sub(r"\b"+str(current_id)+r"\b", local_string, full_string)
            ids_to_replace.pop(0)

        number_subs = 1
        while number_subs == 1:
            full_string, number_subs = re.subn(r"\bi\b", Utilities.get_random_element(['x','y','z'])[0], full_string, count=1)

        return full_string

# if __name__=="__main__":

#     # Utilities.set_seed(35)
#     # Utilities.set_seed(32)
#     Utilities.set_seed(None)

#     ds=None

#     for i in range(10000):
#         # print()
#         if i%1000==0:
#             print(i)
#         Utilities.set_seed(i)
#         # Utilities.set_seed(i+int(1e8)) # For validation
#         my_generator = RandomEquationGenerator(grammar_list, alphabet_with_greek)
#         tree = my_generator.generate_sequence([2,6])
#         variables_symbols = my_generator.assign_variables(tree)
#         latex_parser = LatexParser(conversions_list, grammar_list)
#         chosen_conversions = latex_parser.choose_random_conversions()
#         latex_string = my_generator.convert_to_latex(tree,variables_symbols, chosen_conversions, parenthesis_table, categories_map)
#         pseudocode_string = my_generator.convert_to_string(tree, variables_symbols)

#         # print(latex_string)
#         # print(pseudocode_string)
#         if ds is None:
#             ds = Dataset.from_dict({"conversations": [[{'role': 'user', 'content': latex_string},{'role': 'assistant', 'content': pseudocode_string}]], "seed": [i]})
#         else:
#             ds = ds.add_item({"conversations": [{'role': 'user', 'content': latex_string},{'role': 'assistant', 'content': pseudocode_string}], "seed": i})

#     print(ds)
#     print(ds[0])
#     print(ds[1])

#     ds.save_to_disk("test_dataset.hf")


if __name__=="__main__":

    for i in range(20):
        Utilities.set_seed(i+int(3e8))

        my_generator = RandomEquationGenerator(grammar_list, alphabet_with_greek)
        tree = my_generator.generate_sequence([2,6])
        variables_symbols = my_generator.assign_variables(tree)
        latex_parser = LatexParser(conversions_list, grammar_list)
        chosen_conversions = latex_parser.choose_random_conversions()
        latex_string = my_generator.convert_to_latex(tree,variables_symbols, chosen_conversions, parenthesis_table, categories_map)
        pseudocode_string = my_generator.convert_to_string(tree, variables_symbols)

        # print(tree.upwards_traversal_order)
        # print("test_latex = '"+latex_string+"'")
        # print("test_pseudocode = '"+pseudocode_string+"'")
        print('$'+re.sub(r'\\\\', r'\\', latex_string)+'$\\\\')
        print(re.sub(r'_', r'\\_', pseudocode_string))
        print()