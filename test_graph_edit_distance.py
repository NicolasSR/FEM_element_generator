import networkx as nx
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

def generate_tree(tokens_list):
    tree = nx.DiGraph()

    current_node = 0
    tree.add_node(current_node, value=tokens_list[current_node])
    next_edge_side = 'L'
    current_parent = 0
    pending_bifurcations = [0]
    for val in tokens_list[1:]:
        if val == ')': # Close_parenthesis
            pending_bifurcations.pop()
        elif val == ',': # Comma
            pending_bifurcations.pop()
            current_parent = pending_bifurcations[-1]
            next_edge_side = 'R'
        else:
            current_node+=1
            tree.add_node(current_node, value=val)
            tree.add_edge(current_parent, current_node, side=next_edge_side)
            current_parent=current_node
            pending_bifurcations.append(current_node)
            next_edge_side = 'L'

    print(pending_bifurcations)

    return tree


## Result from LLM will be a vector with the token IDs:

"""
# Ground truth:
# ['inner(prod(var0_0,var4_0),partial(fun(plus(var3_0,var3_1),var1_0),coord_2))']

# ['inner(prod(var0_0,var4_0),partial(plus(fun(var2_0,var3_0),var1_1),var1_0))']
# ['inner(prod(var0_0,var4_0),partial(plus(fun(plus(var3_0,var3_1),var1_0),coord_2),coord_2))']

tokens_list_true = [1,2,100,99,140,98,99,3,5,4,130,99,131,98,99,110,98,99,202,98,98]

tokens_list_1 = [1,2,100,99,140,98,99,3,4,5,120,99,130,98,99,111,98,99,110,98,98]
tokens_list_2 = [1,2,100,99,140,98,99,3,4,5,4,130,99,131,98,99,110,98,99,202,98,99,202,98,98]
"""

# tokens_list_true = ['prod(','var0_0',',','frac(','sign(','var0_1',')',',','sign(','var0_2',')',')',')']
tokens_list_true = ['prod(','var0_0',',','frac(','sign(','var0_1',')',',','sign(','var1_2',')',')',')']

tokens_list_1 = ['prod(','var0_0',',','frac(','sign(','var0_1',')',',','sign(','var0_0',')',')',')']
tokens_list_2 = ['prod(','var0_0',',','frac(','sign(','var0_0',')',',','frac(','sign(','var0_1',')',',','sign(','var0_0',')',')',')',')']


tree_true = generate_tree(tokens_list_true)
tree_1 = generate_tree(tokens_list_1)
tree_2 = generate_tree(tokens_list_2)


# a = nx.DiGraph()
# b = nx.DiGraph()

# a.add_edges_from([('0','1',{'side':'L'}),('0','2',{'side':'R'}),('1','3',{'side':'L'}),('3','4',{'side':'L'}),('1','5',{'side':'R'})])
# # b.add_edges_from([('0','1',{'side':'L'}),('0','2',{'side':'R'}),('1','3',{'side':'L'}),('3','4',{'side':'L'})])
# b.add_edges_from([('0','1',{'side':'L'}),('0','2',{'side':'R'}),('1','5',{'side':'L'}),('1','3',{'side':'R'}),('3','4',{'side':'L'})])
# attrs_a = {'0': {"value": 'prod'}, '1': {"value": 'exp'}, '2': {"value": 'cos'}, '3': {"value": 'contr'},
#            '4': {"value": 'prod'}, '5': {"value": 'cos'}}
#         #    '6': {"value": 'fun'}, '7': {"value": 'contr'},
#         #    '8': {"value": 'dot'}, '9': {"value": 'frac'}, '10': {"value": 'fun'}, '11': {"value": 'sin'}}
# nx.set_node_attributes(a, attrs_a)
# attrs_b = {'0': {"value": 'prod'}, '1': {"value": 'exp'}, '2': {"value": 'cos'}, '3': {"value": 'contr'},
#            '4': {"value": 'prod'}, '5': {"value": 'cos'}}
#         #    '5': {"value": 'sin'}, '6': {"value": 'fun'}, '7': {"value": 'contr'},
#         #    '8': {"value": 'dot'}, '9': {"value": 'frac'}, '10': {"value": 'fun'}, '11': {"value": 'cos'}}
# nx.set_node_attributes(b, attrs_b)

# print(a)
# print(b)

# print(nx.is_arborescence(a))
# print(nx.is_arborescence(b))

def node_subst_fun(node1, node2):
    if node1['value']==node2['value']:
        return 0.0
    elif 'var' in node1['value'] and 'var' in node2['value']:
        if node1['value'][:-1]==node2['value'][:-1]:
            return 0.5
        else:
            return 1.0
    elif 'var' in node1['value'] or 'var' in node2['value']:
        return 1.5
    elif 'coord_' in node1['value'] and 'coord_' in node2['value']:
        return 0.5
    elif 'coord_' in node1['value'] or 'coord_' in node2['value']:
        return 1.5
    # elif 'var' in node1['value']:
    #     if 'var' not in node2['value']:
    #         return 1.5
    #     elif node1['value'][:-1]==node2['value'][:-1]:
    #         return 0.5
    #     else:
    #         return 1.0
    # elif 'coord_' in node1['value']:
    #     if 'var' not in node2['value']:
    #         return 1.5
    #     elif node1['value'][:-1]==node2['value'][:-1]:
    #         return 0.5
    #     else:
    #         return 1.0
    else:
        return 1.0
def node_ins_del_fun(node1):
    return 1.0

def edge_subst_fun(edge1, edge2):
    if edge1['side']==edge2['side']:
        return 0.0
    else:
        return 1.0
def edge_ins_del_fun(edge1):
    return 1.0

edit_distance_1 = nx.graph_edit_distance(tree_true,tree_1, roots=(0,0),
                             node_subst_cost=node_subst_fun,
                             node_del_cost=node_ins_del_fun, node_ins_cost=node_ins_del_fun,
                             edge_subst_cost=edge_subst_fun, 
                             edge_del_cost=edge_ins_del_fun, edge_ins_cost=edge_ins_del_fun)

edit_distance_2 = nx.graph_edit_distance(tree_true,tree_2, roots=(0,0),
                             node_subst_cost=node_subst_fun,
                             node_del_cost=node_ins_del_fun, node_ins_cost=node_ins_del_fun,
                             edge_subst_cost=edge_subst_fun, 
                             edge_del_cost=edge_ins_del_fun, edge_ins_cost=edge_ins_del_fun)


labels_true = nx.get_node_attributes(tree_true, 'value')
edge_labels_true = nx.get_edge_attributes(tree_true,'side')
labels_1 = nx.get_node_attributes(tree_1, 'value')
edge_labels_1 = nx.get_edge_attributes(tree_1,'side')
labels_2 = nx.get_node_attributes(tree_2, 'value')
edge_labels_2 = nx.get_edge_attributes(tree_2,'side')

subax1 = plt.subplot(131)
pos = graphviz_layout(tree_true, prog="dot")
nx.draw(tree_true, pos, labels=labels_true, with_labels=True, font_weight='bold')
nx.draw_networkx_edge_labels(tree_true, pos, edge_labels = edge_labels_true)
plt.title('Ground truth')
subax1 = plt.subplot(132)
pos = graphviz_layout(tree_1, prog="dot")
nx.draw(tree_1, pos, labels=labels_1, with_labels=True, font_weight='bold')
nx.draw_networkx_edge_labels(tree_1, pos, edge_labels = edge_labels_1)
plt.title(f'Edit distance: {edit_distance_1}')
subax3 = plt.subplot(133)
pos = graphviz_layout(tree_2, prog="dot")
nx.draw(tree_2, pos, labels=labels_2, with_labels=True, font_weight='bold')
nx.draw_networkx_edge_labels(tree_2, pos, edge_labels = edge_labels_2)
plt.title(f'Edit distance: {edit_distance_2}')
plt.show()