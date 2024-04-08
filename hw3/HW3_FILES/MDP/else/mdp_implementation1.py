from copy import deepcopy
import numpy as np
from mdp import MDP
from typing import List, Tuple, Dict
import io
import sys

def successors(pos, mdp : MDP):
    """
    returns a list of tuple of kind (string, (int, int))
    empty list in case of a terminal state
    """
    pos_successors = []

    if pos in mdp.terminal_states:
        return pos_successors
    
    for a in mdp.actions:
        pos_successors.append((a, mdp.step(pos, a)))
    return pos_successors
    

def value_iteration_sum(action, Util, pos, mdp : MDP):
    sum = 0
    # wall?
    if mdp.board[pos[0]][pos[1]] == "WALL":
        return -np.inf
    
    for s in successors(pos, mdp):
        sum += mdp.transition_function[action][list(mdp.actions.keys()).index(s[0])] \
            * mdp.gamma * Util[s[1][0]][s[1][1]]
    return sum + float(mdp.board[pos[0]][pos[1]]) # adding reward(curr_pos)

def max_utility(mdp : MDP, U, pos):
    """
    returns None when state is Wall or Terminal
    """
    U_new_r_c = -np.inf
    for a in mdp.actions:
        curr_action_utility = value_iteration_sum(a,
                                                U,
                                                (pos[0],pos[1]),
                                                mdp
                                                )
        if U_new_r_c < curr_action_utility:
            U_new_r_c = curr_action_utility
    return U_new_r_c
    
def value_iteration(mdp : MDP, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    reached_minimal_error = False
    U_final = deepcopy(U_init)
    U_tag = deepcopy(U_init)

    while not reached_minimal_error:
        U_final = deepcopy(U_tag)
        delta = 0
        # loop on each position and calculate it's new Utility
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                if mdp.board[r][c] == "WALL":
                    U_tag[r][c] = "WALL"
                    continue
                U_tag[r][c] = max_utility(mdp, U_final, (r,c))

                # update U' and update delta
                diff = abs(U_tag[r][c] - U_final[r][c])
                if diff > delta:
                    delta = abs(U_tag[r][c] - U_final[r][c])
        
        if delta < epsilon*(1-mdp.gamma)/mdp.gamma:
            reached_minimal_error = True
    return U_final

    # ========================
def wall_or_terminal(mdp : MDP, pos : Tuple[int]):
    if pos in mdp.terminal_states or mdp.board[pos[0]][pos[1]] == "WALL":
        return True
    return False

def get_best_action(U, pos, mdp : MDP):
    """
    returns None in case of terminal or WALL
    """

    max_action_utility = (None, -np.inf)

    for a in mdp.actions:

        if wall_or_terminal(mdp, pos):
            break

        curr_action_utility = value_iteration_sum(a, U, pos, mdp)
        if max_action_utility[1] < curr_action_utility:
            max_action_utility = (a, curr_action_utility)
    return max_action_utility[0]


def get_policy(mdp : MDP, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    # ====== YOUR CODE: ======
    # For each position, calculate using the bellman-equation, which action has the max utility!
    policy = [[None for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]

    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            policy[r][c] = get_best_action(U, (r,c), mdp)
    
    return policy
    # ========================

def get_probs_for_s(mdp : MDP, pos, policy_action):
    """
    helper function for solving policy evaluation linear system
    guess what it does
    """
    probs = []
    if policy_action == None:
        return probs
    
    next_states = successors(pos, mdp)
    for ns in next_states:
        r = ns[1][0]
        c = ns[1][1]
        action = ns[0]
        prob_to_reach_ns_when_executing_policy_action = \
            mdp.transition_function[policy_action][list(mdp.actions.keys()).index(action)]
        probs.append((r*mdp.num_col+c, mdp.gamma*prob_to_reach_ns_when_executing_policy_action))
    return probs

def x_to_utility(mdp : MDP, x):
    U = [[0 for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            U[r][c] = x[r*mdp.num_col+c] if mdp.board[r][c] != "WALL" else None
    return U

def policy_evaluation(mdp : MDP, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    # given a policy, use bellman equation to form a linear equation system with n variables
    # when n is the number of states
    # x will be the answer vector
    # b will be R(s) values
    # A will be: a matrix that:
    #               - a row represents one linear equation that have at most 4 variables:
    #                           (1) U(s)
    #                           (2...) U((s' |a_i))
    # e.g, A = [first equation (of U(s_1)): [1, ...-p(s? |pi[a])]]
    n = mdp.num_row * mdp.num_col
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    b = [0 for _ in range(n)]

    for r in range(mdp.num_row):
        for c in range( mdp.num_col):
            curr_equation_number = r*mdp.num_col+c
            if mdp.board[r][c] == "WALL":
                matrix[curr_equation_number][curr_equation_number]
            
            for param in get_probs_for_s(mdp, (r,c), policy[r][c]):
                matrix[curr_equation_number][param[0]] += param[1]
            matrix[curr_equation_number][curr_equation_number] -= 1.0 # u((r,c)) = R((r,c)) + segma... => -R((r,c)) = -u((r,c)) + segma...
            # b = (R(s_0_0), ...) |b| = n
            b[curr_equation_number] = (-1.0) * float(mdp.board[r][c]) if\
                  mdp.board[r][c] != "WALL" else -1.0 #-R((r,c)) = ...

    # x = A^-1 * b
    A = np.array(matrix)
    bb = np.array(b)
    x = np.linalg.solve(A, bb)
    # return x
    return x_to_utility(mdp, x)
    # ========================


def policy_iteration(mdp : MDP, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    optimal_policy = deepcopy(policy_init)    # should deepcopy be used here?
    changed = True

    while changed:
        U = policy_evaluation(mdp, optimal_policy)
        changed = False
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                if value_iteration_sum(optimal_policy[r][c], U, (r,c), mdp) <\
                max_utility(mdp, U, (r,c)):
                    prev_policy_action = optimal_policy[r][c]
                    optimal_policy[r][c] = get_best_action(U, (r,c), mdp)
                    if optimal_policy[r][c] != prev_policy_action:
                        changed = True

    return optimal_policy

    # ========================



"""For this functions, you can import what ever you want """

def utilEqual(util1, util2, epsilon):
    x = len(str(1/epsilon).split(".")[0])+1
    return abs(round(util1, x) - round(util2, x)) <= epsilon

def getActionSymbol(action):
    """
    e.g., action == UP, return U
    """
    return action[0]

def get_equal_actions(action, U, pos, mdp : MDP, epsilon):
    """
    returns None in case of terminal or WALL
    """
    if wall_or_terminal(mdp, pos):
        return None
    
    # max_action_utility = U[pos[0]][pos[1]]
    max_action_utility = value_iteration_sum(action, U, pos, mdp)

    # if round(max_action_utility, 3) != round(U[pos[0]][pos[1]], 3):
    #     print(round(max_action_utility, 4), " : ", round(U[pos[0]][pos[1]], 4))
    #     # in our case, the action passed as an arg has to be the best action
    #     raise RuntimeError("What!!! why not equal\n")
    
    similar_actions = []

    for a in mdp.actions:

        curr_action_utility = value_iteration_sum(a, U, pos, mdp)
        if utilEqual(max_action_utility, curr_action_utility, epsilon):
            similar_actions.append(getActionSymbol(a))

    return similar_actions

def get_similar_policies(mdp : MDP, policy, U, epsilon):
    policy_merged = [[None for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]

    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            policy_merged[r][c] = get_equal_actions(policy[r][c], U, (r,c), mdp, epsilon)
    return policy_merged

def number_of_different_policies(n, p, merged_policy):
    """
    e.g., 
    policy :
    |UD|UL|
    |DU|URL|

    returns 2*2*2*3 = 24
    """
    count = 1
    for r in range(n):
        for c in range(p):
            actions = merged_policy[r][c]
            if actions != None:
                count *= len(actions)
    return count

def get_all_policies(mdp : MDP, U, epsilon=10 ** (-3)):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp, and the utility value U (which satisfies the Belman equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies
    #

    # ====== YOUR CODE: ======
    best_policy = get_policy(mdp, U)
    merged_policy = get_similar_policies(mdp, best_policy, U, epsilon)

    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            if merged_policy[r][c] is None:
                continue
            merged_policy[r][c] = "".join(merged_policy[r][c])

    mdp.print_policy(merged_policy)

    return number_of_different_policies(mdp.num_row, mdp.num_col, merged_policy)
    # ========================


def update_board(mdp : MDP, add):
    for r in range(len(mdp.board)):
        for c in range(len(mdp.board[0])):
            if (r, c) in mdp.terminal_states or mdp.board[r][c] == "WALL":
                continue
            mdp.board[r][c] = add

def parse_table(output):
    rows = output.strip().split('\n')
    matrix = [list(map(str.strip, row.strip().split('|')[1:-1])) for row in rows]
    return matrix

def same_sets(str1, str2):
    """
    returns 1 if both strings have the same histogram
    """
    # Create histograms of letter counts for both strings
    hist1 = {}
    hist2 = {}

    for char in str1:
        hist1[char] = hist1.get(char, 0) + 1

    for char in str2:
        hist2[char] = hist2.get(char, 0) + 1

    # Compare the histograms
    if hist1 == hist2:
        return 0
    else:
        return 1
    
def comparePolicies_with_parsing(p1, p2):
    """
    return 1 if tables differ
    """
    p1_parsed = parse_table(p1)
    p2_parsed = parse_table(p2)

        # Compare the tables
    for i in range(len(p1_parsed)):
        for j in range(len(p1_parsed[i])):
            if not same_sets(p1_parsed[i][j], p2_parsed[i][j]):
                return 1
    return 0

def comparePolicies(p1, p2):
    """
    return 1 if tables differ
    """
    return 1 if p1 != p2 else 0


def get_policy_for_different_rewards(mdp : MDP):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp
    # print / displas the optimal policy as a function of r
    # (reward values for any non-finite state)
    #

    # ====== YOUR CODE: ======
        # round Utilities

    # plan:
    # for each 0.01 from -5 to 5 do:
    # define a new reward function:
    #   +0.01 for each s that is not terminal or dead end
    # calculate U using the new reward function with value_iteration
    # get_policy for U
    # get_all_policies similiar to get_policy result
    # check if last get_all_policies result of the last U differs from the current get_all_policies result
    # if so add a new range

    # q: what can you do to make it quick
    # what are the ranges we know the policy won't change within
    # maybe for all R>0 the policies will be the same

    reward_ranges = []
    uniformal_reward = -5.0
    update_board(mdp, uniformal_reward)
    

    stdout = sys.stdout # used because of the fact that get_all_polycies prints and does not return a table!
    last_policy = None

    try:
        while uniformal_reward <= 5:
            Utility = value_iteration(mdp, [[0 for _ in range(mdp.num_col)] for _ in range(mdp.num_row)])
            # get all poicies

            sys.stdout = io.StringIO()

            get_all_policies(mdp, Utility)
            newPolicy = sys.stdout.getvalue()

            if last_policy != None:
                # sys.stdout = stdout
                # print(last_policy)
                # print(newPolicy)
                if comparePolicies(last_policy, newPolicy):
                    reward_ranges.append(round(uniformal_reward, 3))
                    sys.stdout = stdout
                    if len(reward_ranges) == 1:
                        print("R(s) < ", reward_ranges[-1])
                    else:
                        print(reward_ranges[-2], " <= R(s) < ", reward_ranges[-1])
                    mdp.print_policy(parse_table(last_policy))


            # last step
            uniformal_reward += 0.01
            update_board(mdp, uniformal_reward)
            last_policy = deepcopy(newPolicy)
    finally:
        sys.stdout = stdout
        print(reward_ranges[-1], "<= R(s)")
        mdp.print_policy(parse_table(newPolicy))
    return reward_ranges
    # ========================
