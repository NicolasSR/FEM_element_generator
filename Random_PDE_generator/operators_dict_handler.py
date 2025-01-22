from collections import OrderedDict

import numpy as np

from seeded_random_generator import SeededRandomGenerator as Utilities
from tree_nodes import UnaryOperatorNode, BinaryOperatorNode, PartialUnaryOperatorNode, PolyOperatorNode, ArggroupNode

# This is a dataset containing the specific notation to be used by our LLM in order to generate pseudocode for PDEs
# It includes the operator notation, the input rank limitations, the output rank (as afunction of the input), and a description
# We are targeting only variational forms of PDEs in the context of FEM. Therefore, the dimensions will be fixed by the domain
#   so we don't need to care about them

operators_dict= OrderedDict((
    ("unary", {
        "class": UnaryOperatorNode,
        "grammar": OrderedDict((
            # Scalar specific
            ("sqrt(a)", {
                "out_relation": 'a',
                "in_ranks": [0],
                "out_ranks": [0],
                "latex": [[':/:sqrt{a}', 'parenthesis']],
                "info": "square root of a"
                }),
            ("exp(a)", {
                "out_relation": 'a',
                "in_ranks": [0],
                "out_ranks": [0],
                "latex": [['e^{a}', 'exp_R'], [':/:exp{a}', 'mono']],
                "info": "exponential of scalar"
                }),
            ("ln(a)", {
                "out_relation": 'a',
                "in_ranks": [0],
                "out_ranks": [0],
                "latex": [[':/:ln{a}', 'mono']],
                "info": "natural log of scalar"
                }),
            ("abs(a)", {
                "out_relation": 'a',
                "in_ranks": [0],
                "out_ranks": [0],
                "latex": [[':/:text{abs}:_:a','mono'], ['|a|', 'parenthesis']],
                "info": "absolute value of scalar"
                }),
            ("sign(a)", {
                "out_relation": 'a',
                "in_ranks": [0],
                "out_ranks": [0],
                "latex": [[':/:text{sgn}:_:a', 'mono'], [':/:text{sign}:_:a', 'mono']],
                "info": "sign of scalar (1 or -1)"
                }),
            # Matrix specific
            ("transpose(a)", {
                "out_relation": 'a',
                "in_ranks": [2],
                "out_ranks": [2],
                "latex": [['a^T', 'exp_L']],
                "info": "a^T"
                }),
            ("sym(a)", {
                "out_relation": 'a',
                "in_ranks": [2],
                "out_ranks": [2],
                "latex": [[':/:text{sym}:_:a', 'mono']], # :/:sym{a}
                "info": "a=1/2(a+a^T)"
                }),
            ("skew(a)", {
                "out_relation": 'a',
                "in_ranks": [2],
                "out_ranks": [2],
                "latex": [[':/:text{skew}:_:a', 'mono']], # :/:skew{a}
                "info": "a=1/2(a-a^T)"
                }),
            ("dev(a)", {
                "out_relation": 'a',
                "in_ranks": [2],
                "out_ranks": [2],
                "latex": [[':/:text{dev}:_:a', 'mono']], # :/:dev{a}
                "info": "deviatoric part of a (check Cauchy Stress)"
                }),
            ("tr(a)", {
                "out_relation": '0',
                "in_ranks": [2],
                "out_ranks": [0],
                "latex": [[':/:text{Tr}:_:a', 'mono']], # :/:Tr{a}
                "info": "trace of matrix"
                }),
            ("det(a)", {
                "out_relation": '0',
                "in_ranks": [2],
                "out_ranks": [0],
                "latex": [[':/:text{det}:_:a', 'mono'], [':/:left| a :/:right|', 'parenthesis']], # :/:det{a}
                "info": "determinant of matrix"
                }),
            # Differential operators
            ("grad_n(a)", {
                "out_relation": 'a',
                "in_ranks": [0],
                "out_ranks": [0],
                "latex": [[':/:nabla_n a', 'mono']],
                "info": "Dot product of gradient of scalar a with normal vector"
                }),
            ("grad(a)", {
                "out_relation": 'a+1',
                "in_ranks": [0],
                "out_ranks": [1],
                "latex": [[':/:nabla a', 'mono']],
                "info": "gradient of a"
                }),
            ("div(a)", {
                "out_relation": 'a-1',
                "in_ranks": [1],
                "out_ranks": [0],
                "latex": [[':/:nabla :/:cdot a', 'poly_tens_R']],
                "info": "divergence of a"
                }),
            ("curl(a)", {
                "out_relation": 'a',
                "in_ranks": [1],
                "out_ranks": [1],
                "latex": [[':/:nabla :/:times a', 'poly_tens_R']],
                "info": "curl of a"
                })))
            }
        ),
    ("binary", {
        "class": BinaryOperatorNode,
        "grammar": OrderedDict((
            # Scalar specific
            ("pow(a,b)", {
                "in_relation": 'a==b==0',
                "out_relation": 'a',
                "in_ranks": [[0],[0]],
                "out_ranks": [0],
                "latex": [['a^{b}', 'exp']],
                "info": "a^b"
                }),
            # Semi-Scalar specific
            ("prod(a,b)", {
                "in_relation": 'min(a,b)==0',
                "out_relation": 'max(a,b)',
                "in_ranks": [[0,1,2,3,4],[0,1,2,3,4]],
                "out_ranks": [0,1,2,3,4],
                "latex": [['a b', 'poly_prod'], ['a :/:cdot b', 'poly_prod']],
                "info": "ab, aB, Ab..."
                }),
            ("frac(a,b)", {
                "in_relation": 'b==0',
                "out_relation": 'a',
                "in_ranks": [[0,1,2,3,4],[0]],
                "out_ranks": [0,1,2,3,4],
                "latex": [[':/:frac{a}{b}', 'frac']],
                "info": "a/b"
                }),
            # Vector specific
            ("dot(a,b)", {
                "in_relation": 'a==b==1',
                "out_relation": 'a-b',
                "in_ranks": [[1],[1]],
                "out_ranks": [0],
                "latex": [['a :/:cdot b', 'poly_tens']],
                "info": "dot product of 2 vectors"
                }),
            ("cross(a,b)", {
                "in_relation": 'a==b==1',
                "out_relation": 'a',
                "in_ranks": [[1],[1]],
                "out_ranks": [1],
                "latex": [['a :/:times b', 'poly_tens']],
                "info": "cross product of 2 vectors"
                }),
            # Matrix specific
            ("matrix_prod(a,b)", {
                "in_relation": 'a==b==2',
                "out_relation": 'a',
                "in_ranks": [[2],[2]],
                "out_ranks": [2],
                "latex": [['a b', 'poly_tens']],
                "info": "classic matrix product: AB_ik = A_ij*B_jk"
                }),
            # General
            ("plus(a,b)", {
                "in_relation": 'a==b',
                "out_relation": 'a',
                "in_ranks": [[0,1,2,3,4],[0,1,2,3,4]],
                "out_ranks": [0,1,2,3,4],
                "latex": [['a+b', 'poly_sum']],
                "info": "a+b"
                }),
            ("outer(a,b)", {
                "in_relation": 'min(a,b)>0',
                "out_relation": 'a+b',
                "in_ranks": [[1,2,3],[1,2,3]],
                "out_ranks": [2,3,4],
                "latex": [['a :/:otimes b', 'poly_tens']],
                "info": "a+b"
                }),
            ("contract(a,b)", {
                "in_relation": 'min(a,b)>0',
                "out_relation": 'max(a,b)-min(a,b)',
                "in_ranks": [[1,2,3,4],[1,2,3,4]],
                "out_ranks": [0,1,2,3],
                "latex": [['a:b', 'poly_tens']],
                "info": "cotract tensors: c_i=a_ijk*b_jk, c_k=a_ij*b_ijk"
                }),
            ("inner(a,b)", {
                "in_relation": 'a==b>0',
                "out_relation": 'a-b',
                "in_ranks": [[1,2,3,4],[0,1,2,3,4]],
                "out_ranks": [0],
                "latex": [['a :/:cdot b', 'poly_tens'], [':/:langle a,b :/:rangle', 'parenthesis']],
                "info": "<a,b> = a_ij*b_ij"
                })))
            }
        ),
    ("partial_unary", {
        "class": PartialUnaryOperatorNode,
        "grammar": OrderedDict((
            # Differential operator, i represents the component with respect to which we differenciate
            ("partial(a,i)", {
                "out_relation": 'a',
                "in_ranks": [0,1,2,3,4],
                "out_ranks": [0,1,2,3,4],
                "latex": [[':/:frac{:/:partial a}{:/:partial i}', 'mono'], [':/:partial_{i}a', 'mono']],
                "info": "Partial derivative over one component of the domain space"
                }),))
            }
        ),
    ("poly", {
        "class": PolyOperatorNode,
        "grammar": OrderedDict((
            # User-defined functions
            ("fun(a)", {
                "in_ranks": [0,1,2,3,4],
                "out_ranks": [0,1,2,3,4],
                "latex": [[':/:text{fun}:_:a', 'mono']],
                "info": "Generic, user-defined function"
                }),)),
            }
        ),
    ("arggroup", {
        "class": ArggroupNode,
        "grammar": OrderedDict((
            # [pseudocode grammar, description]
            ("a,b", {
                "in_ranks": [[0,1,2,3,4],[0,1,2,3,4]],
                "latex": [['a,b', 'punctuation']],
                "info": "Argument group. To be able to add more arguments into user-defined functions. Cannot be selected randomly"
                }),)),
            }
        )
    ))

parenthesis_table = None
#### WE HAVE AN ERROR: If a poly_tens enters a product on the right and the product's left child is not a scalar, we need to have paarenthesis on the right operator child.
# Example: \boldsymbol{\psi} \boldsymbol{\psi} \cdot \boldsymbol{\psi} \cdot \text{fun}(\frac{\boldsymbol{\psi}}{\varphi},\boldsymbol{\Xi},\varphi)$\\
#           dot(prod(psi\_bold,dot(psi\_bold,psi\_bold)),fun(frac(psi\_bold,varphi),Xi\_bold,varphi))


## Parenthesis_table:
parenthesis_table = np.array([
     # poly_sum_L | poly_sum_R | poly_tens_L | poly_tens_R | poly_prod_L | poly_prod_R | exp_L | exp_R | mono | frac | parenthesis | punctuation
    [   0,              0,          1,              1,          1,          1,          1,      0,      1,      0,      0,           0   ],      #poly_sum
    [   0,              0,          0,              1,          0,          2,          1,      0,      1,      0,      0,           0   ],      #poly_tens
    [   0,              0,          0,              0,          0,          0,          1,      0,      1,      0,      0,           0   ],      #poly_prod
    [   0,              0,          0,              0,          0,          0,          1,      0,      0,      0,      0,           0   ],      #exp
    [   0,              0,          0,              0,          0,          0,          1,      0,      0,      0,      0,           0   ],      #mono
    [   0,              0,          0,              0,          0,          0,          1,      0,      0,      0,      0,           0   ],      #frac
    [   0,              0,          0,              0,          0,          0,          0,      0,      0,      0,      0,           0   ],      #parenthesis
    [   0,              0,          0,              0,          0,          0,          0,      0,      1,      0,      0,           0   ]       #punctuation
    ])

precedence_map = {
    "in_type":  {
        'poly_sum':0,
        'poly_tens':1,
        'poly_tens_R':1,
        'poly_prod':2,
        'exp':3,
        'exp_L':3,
        'exp_R':3,
        'mono':4,
        'frac':5,
        'parenthesis':6,
        'punctuation':7
        },
    "slot_type":{
        'poly_sum_L':0,
        'poly_sum_R':1,
        'poly_tens_L':2,
        'poly_tens_R':3,
        'poly_prod_L':4,
        'poly_prod_R':5,
        'exp_L':6,
        'exp_R':7,
        'mono_L':8,
        'mono_R':8,
        'frac_L':9,
        'frac_R':9,
        'parenthesis_L':10,
        'parenthesis_R':10,
        'punctuation_L':11,
        'punctuation_R':11
        }
    }

class OperatorDictHandler():
    
    def __init__(self, operators_dict, parenthesis_table, precedence_map):
        self.operators_dict = operators_dict
        self.parenthesis_table = parenthesis_table
        self.precedence_map = precedence_map

        self._generate_rank_tables()

        self.choose_random_conversions()
    
    def reset(self):
        self.choose_random_conversions()

    def _generate_rank_tables(self):
        for key, value in self.operators_dict.items():
            rank_table = value["class"].get_rank_table(value["grammar"])
            self.operators_dict[key]["rank_table"]=rank_table

    def get_compatible_operators_by_rank(self, rank, compatible_children_groups):
        mask = []
        operators_list = []
        for group in compatible_children_groups:
            mask.append(self.operators_dict[group]["rank_table"][:,rank].astype(bool))
            operators_list.append([oper for oper in self.operators_dict[group]["grammar"].keys()])
        mask = np.concatenate(mask)
        operators_list = np.concatenate(operators_list)
        return operators_list[mask]
    
    def get_class_from_operator(self, query_operator):
        for value in self.operators_dict.values():
            if query_operator in value["grammar"].keys():
                return value["class"]
        raise ValueError(f"Operator {query_operator} was not found in the operator dictionary")

    def get_info_from_operator(self, query_operator):
        for value in self.operators_dict.values():
            if query_operator in value["grammar"].keys():
                return value["grammar"][query_operator]
        raise ValueError(f"Operator {query_operator} was not found in the operator dictionary")
    
    def get_unique_operator_from_class(self, query_class):
        for value in self.operators_dict.values():
            if value["class"] == query_class:
                operators_list = [oper for oper in value["grammar"].keys()]
                assert len(operators_list)==1, f"Grammar list for queried class {query_class} has more than one entry"
                return operators_list[0]
        raise ValueError(f"Queried class {query_class} was not found in the grammar dictionary")
    
    def choose_random_conversions(self):
        self.current_latex_notations = {}
        for value in self.operators_dict.values():
            for operator, operator_info in value["grammar"].items():
                chosen_form = Utilities.get_random_element(operator_info["latex"])[0]
                self.current_latex_notations[operator] = chosen_form

    def get_operator_latex(self, operator):
        return self.current_latex_notations[operator]
