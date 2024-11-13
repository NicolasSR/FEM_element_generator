import numpy as np

from Random_PDE_generator.seeded_random_generator import SeededRandomGenerator as Utilities

conversions_list=[
# Scaler specific
['pow(a,b)', ['a^{b}', 'exp']],
['sqrt(a)', [':/:sqrt{a}', 'parenthesis']],
['exp(a)', ['e^{a}', 'exp_R'], [':/:exp{a}', 'mono']],
['ln(a)', [':/:ln{a}', 'mono']],
['abs(a)', [':/:abs{a}','mono'], ['|a|', 'parenthesis']],
['sign(a)', [':/:sgn{a}', 'mono'], [':/:sign{a}', 'mono']], # Needs to be implemented in Latex: \DeclareMathOperator{\sgn}{}

# Semi-Scalar specific
['prod(a,b)', ['a b', 'poly_prod'], ['a :/:cdot b', 'poly_prod']],
['frac(a,b)', [':/:frac{a}{b}', 'frac']],

# Vector specific
['dot(a,b)', ['a :/:cdot b', 'poly_tens']],
['cross(a,b)', ['a :/:times b', 'poly_tens']],

# Matrix specific
['matrix_prod(a,b)', ['a b', 'poly_tens']],
['transpose(a)', ['a^T', 'exp_L']],
['sym(a)', [':/:sym{a}', 'mono']], # Needs to be implemented in Latex: \DeclareMathOperator{\sym}{}
['skew(a)', [':/:skew{a}', 'mono']], # Needs to be implemented in Latex: \DeclareMathOperator{\skew}{}
['dev(a)', [':/:dev{a}', 'mono']], # Needs to be implemented in Latex: \DeclareMathOperator{\dev}{}
['tr(a)', [':/:Tr{a}', 'mono']], # Needs to be implemented in Latex: \DeclareMathOperator{\Tr}{Tr}
['det(a)', [':/:det{a}', 'mono'], [':/:left| a :/:right|', 'parenthesis']], # Needs to be implemented in Latex: \DeclareMathOperator{\det}{}

# General
['plus(a,b)', ['a+b', 'poly_sum']],  
['outer(a,b)', ['a :/:otimes b', 'poly_tens']], # Needs to be implemented in Latex: \DeclareMathOperator{\prodouter}{\otimes}
['contract(a,b)', ['a:b', 'poly_tens']],
['inner(a,b)', ['a :/:cdot b', 'poly_tens'], [':/:langle a,b :/:rangle', 'parenthesis']],

# Differential operators
['partial(a,i)', [':/:frac{:/:partial a}{:/:partial i}', 'frac'], [':/:partial_{i}a', 'mono']],
# ['der_n(a)', [':/:nabla a :/:cdot n', 'poly_tens'], ['n :/:cdot :/:nabla a', 'poly_tens']],  # This might need some particularities in terms of parenthesis
['der_n(a)', [':/:nabla_n a', 'mono']],
['grad(a)', [':/:nabla a', 'mono']],
['div(a)', [':/:nabla :/:cdot a', 'poly_tens']],
['curl(a)', [':/:nabla :/:times a', 'poly_tens']],

# User-defined functions
['fun(a)', [':/:text{fun}(a)', 'parenthesis']], # 'fun' needs to be replaced by random strings
['a,b', ['a,b', 'parenthesis']]
]


## Parenthesis_table:
parenthesis_table = np.array([
     # poly_sum_L | poly_sum_R | poly_tens_L | poly_tens_R | poly_prod | exp_L | exp_R | mono | frac | parenthesis
    [   0,              0,          1,              1,          1,          1,      0,      1,  0,      0],      #poly_sum
    [   0,              0,          0,              1,          0,          1,      0,      1,  0,      0],      #poly_tens
    [   0,              0,          0,              0,          0,          1,      0,      1,  0,      0],      #poly_prod
    [   0,              0,          0,              0,          0,          1,      0,      0,  0,      0],      #exp
    [   0,              0,          0,              0,          0,          1,      0,      0,  0,      0],      #mono
    [   0,              0,          0,              0,          0,          1,      0,      0,  0,      0],      #frac
    [   0,              0,          0,              0,          0,          0,      0,      0,  0,      0]       #parenthesis
])

categories_map = [{'poly_sum':0,'poly_tens':1,'poly_prod':2,'exp':3, 'exp_L':3, 'exp_R':3, 'mono':4,'frac':5,'parenthesis':6},
    {'poly_sum_L':0,'poly_sum_R':1,'poly_tens_L':2,'poly_tens_R':3,'poly_prod_L':4,'poly_prod_R':4,'exp_L':5,'exp_R':6,'mono_L':7,'frac_L':8,'frac_R':8,'parenthesis_L':9,'parenthesis_R':9}]


class LatexParser():
    
    def __init__(self, conversions_list, grammar_list):
        self.conversions_list = conversions_list
        if not self.check_conversions_and_grammar_lists(grammar_list):
            raise Exception("Grammar list and conversion list do not match") 

    def check_conversions_and_grammar_lists(self, grammar_list):
        if not len(grammar_list)==len(self.conversions_list):
            return False
        for gram, conv in zip(grammar_list, conversions_list):
            if not gram[0]==conv[0]:
                return False
        return True
    
    def choose_random_conversions(self):
        chosen_conversions = []
        for conv in conversions_list:
            chosen_form_id = Utilities.get_random_element(list(range(len(conv)-1)))[0]
            chosen_conversions.append(conv[chosen_form_id+1])
        return chosen_conversions