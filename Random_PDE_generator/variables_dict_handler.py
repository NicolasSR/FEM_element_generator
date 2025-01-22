from copy import deepcopy
from collections import OrderedDict

from seeded_random_generator import SeededRandomGenerator as Utilities

variables_dict= OrderedDict((
    # Rank: [[pseudocode,latex],...]
    ('0', [["d", "d"], ["e", "e"], ["f", "f"], ["g", "g"],
    ["h", "h"], ["k","k"], ["l", "l"], ["m", "m"], ["n", "n"],
    ["o", "o"], ["p", "p"], ["q", "q"], ["r", "r"], ["s", "s"], ["t", "t"], ["u", "u"],
    ["v", "v"], ["w", "w"], ["x", "x"], ["y", "y"], ["z", "z"],
    ["alpha", ":/:alpha"], ["beta", ":/:beta"], ["gamma", ":/:gamma"],
    ["delta", ":/:delta"], ["epsilon", ":/:epsilon"], ["zeta", ":/:zeta"],
    ["eta", ":/:eta"], ["theta", ":/:theta"], ["kappa", ":/:kappa"],
    ["lambda", ":/:lambda"], ["mu", ":/:mu"], ["nu", ":/:nu"], ["xi", ":/:xi"],
    ["pi", ":/:pi"], ["rho", ":/:rho"], ["sigma", ":/:sigma"], ["tau", ":/:tau"],
    ["upsilon", ":/:upsilon"], ["phi", ":/:phi"], ["chi", ":/:chi"],
    ["psi", ":/:psi"], ["omega", ":/:omega"], ["varphi", ":/:varphi"],
    ["varepsilon", ":/:varepsilon"]]),

    ('1', [["c_bold", ":/:mathbf{c}"], ["d_bold", ":/:mathbf{d}"], ["e_bold", ":/:mathbf{e}"], ["f_bold", ":/:mathbf{f}"], ["g_bold", ":/:mathbf{g}"],
    ["h_bold", ":/:mathbf{h}"], ["k_bold", ":/:mathbf{k}"], ["l_bold", ":/:mathbf{l}"], ["m_bold", ":/:mathbf{m}"], ["n_bold", ":/:mathbf{n}"],
    ["o_bold", ":/:mathbf{o}"], ["p_bold", ":/:mathbf{p}"], ["q_bold", ":/:mathbf{q}"], ["r_bold", ":/:mathbf{r}"], ["s_bold", ":/:mathbf{s}"], ["t_bold", ":/:mathbf{t}"], ["u_bold", ":/:mathbf{u}"],
    ["v_bold", ":/:mathbf{v}"], ["w_bold", ":/:mathbf{w}"], ["x_bold", ":/:mathbf{x}"], ["y_bold", ":/:mathbf{y}"], ["z_bold", ":/:mathbf{z}"],
    ["alpha_bold", ":/:boldsymbol{:/:alpha}"], ["beta_bold", ":/:boldsymbol{:/:beta}"], ["gamma_bold", ":/:boldsymbol{:/:gamma}"],
    ["delta_bold", ":/:boldsymbol{:/:delta}"], ["epsilon_bold", ":/:boldsymbol{:/:epsilon}"], ["zeta_bold", ":/:boldsymbol{:/:zeta}"],
    ["eta_bold", ":/:boldsymbol{:/:eta}"], ["theta_bold", ":/:boldsymbol{:/:theta}"], ["kappa_bold", ":/:boldsymbol{:/:kappa}"],
    ["lambda_bold", ":/:boldsymbol{:/:lambda}"], ["mu_bold", ":/:boldsymbol{:/:mu}"], ["nu_bold", ":/:boldsymbol{:/:nu}"], ["xi_bold", ":/:boldsymbol{:/:xi}"],
    ["pi_bold", ":/:boldsymbol{:/:pi}"], ["rho_bold", ":/:boldsymbol{:/:rho}"], ["sigma_bold", ":/:boldsymbol{:/:sigma}"], ["tau_bold", ":/:boldsymbol{:/:tau}"],
    ["upsilon_bold", ":/:boldsymbol{:/:upsilon}"], ["phi_bold", ":/:boldsymbol{:/:phi}"], ["chi_bold", ":/:boldsymbol{:/:chi}"],
    ["psi_bold", ":/:boldsymbol{:/:psi}"], ["omega_bold", ":/:boldsymbol{:/:omega}"], ["varphi_bold", ":/:boldsymbol{:/:varphi}"],
    ["varepsilon_bold", ":/:boldsymbol{:/:varepsilon}"]]),

    ('2', [["A", "A"], ["B", "B"], ["C", "C"], ["D", "D"], ["E", "E"], ["F", "F"], ["G", "G"],
    ["H", "H"], ["I", "I"], ["J", "J"], ["K", "K"], ["L", "L"], ["M", "M"], ["N", "N"],
    ["O", "O"], ["P", "P"], ["Q", "Q"], ["R", "R"], ["S", "S"], ["T", "T"], ["U", "U"],
    ["V", "V"], ["W", "W"], ["X", "X"], ["Y", "Y"], ["Z", "Z"]]),

    ('3', [["A_bold", ":/:mathbf{A}"], ["B_bold", ":/:mathbf{B}"], ["C_bold", ":/:mathbf{C}"], ["D_bold", ":/:mathbf{D}"], ["E_bold", ":/:mathbf{E}"], ["F_bold", ":/:mathbf{F}"], ["G_bold", ":/:mathbf{G}"],
    ["H_bold", ":/:mathbf{H}"], ["I_bold", ":/:mathbf{I}"], ["J_bold", ":/:mathbf{J}"], ["K_bold", ":/:mathbf{K}"], ["L_bold", ":/:mathbf{L}"], ["M_bold", ":/:mathbf{M}"], ["N_bold", ":/:mathbf{N}"],
    ["O_bold", ":/:mathbf{O}"], ["P_bold", ":/:mathbf{P}"], ["Q_bold", ":/:mathbf{Q}"], ["R_bold", ":/:mathbf{R}"], ["S_bold", ":/:mathbf{S}"], ["T_bold", ":/:mathbf{T}"], ["U_bold", ":/:mathbf{U}"],
    ["V_bold", ":/:mathbf{V}"], ["W_bold", ":/:mathbf{W}"], ["X_bold", ":/:mathbf{X}"], ["Y_bold", ":/:mathbf{Y}"], ["Z_bold", ":/:mathbf{Z}"]]),

    ('4', [["Gamma_bold", ":/:boldsymbol{:/:Gamma}"], ["Delta_bold", ":/:boldsymbol{:/:Delta}"],
    ["Theta_bold", ":/:boldsymbol{:/:Theta}"], ["Lambda_bold", ":/:boldsymbol{:/:Lambda}"], ["Xi_bold", ":/:boldsymbol{:/:Xi}"],
    ["Pi_bold", ":/:boldsymbol{:/:Pi}"], ["Sigma_bold", ":/:boldsymbol{:/:Sigma}"], ["Upsilon_bold", ":/:boldsymbol{:/:Upsilon}"],
    ["Phi_bold", ":/:boldsymbol{:/:Phi}"], ["Psi_bold", ":/:boldsymbol{:/:Psi}"], ["Omega_bold", ":/:boldsymbol{:/:Omega}"]])
))

class VariableDictHandler():
    
    def __init__(self, variable_dict):
        self.variable_dict = variable_dict
        self.symbol_candidates = deepcopy(variable_dict)

    def reset(self):
        self.symbol_candidates = deepcopy(self.variable_dict)
    
    def select_new_symbol_by_rank(self, rank):
        for key in self.symbol_candidates.keys():
            if str(rank) in key:
                symbol, symbol_idx = Utilities.get_random_element(self.symbol_candidates[key])
                self.symbol_candidates[key].pop(symbol_idx)
        return symbol