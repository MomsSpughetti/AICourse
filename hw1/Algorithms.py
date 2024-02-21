import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

import time
from IPython.display import clear_output


class BFS_node:
    def __init__(self, state: Tuple, parent = None) -> None:
        self.state = state
        self.terminated = False
        self.parent = parent if isinstance(parent, BFS_node) else None
    def __eq__(self, other):
        return isinstance(other, BFS_node) and self.state[0] == other.state[0] and self.state[1] == other.state[1] and self.state[2] == other.state[2]

class Agent():
    def __init__(self) -> None:
        self.expanded = 0
        self.env = None

    def solution(self, node : BFS_node) -> list[int]:
        """
        Finds a path to a goal
        - Parameters: a node of type BFS_node
        - return:
            List[int] as the path
        """
        raise NotImplementedError("Subclass must implement abstract method 'search()'")
    
    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        """
        finds optimal path from s to g
        - return (actions, cost, expanded)
        """
        raise NotImplementedError("Subclass must implement abstract method 'search()'")
    
    def animation(self, state: int) -> None:
        clear_output(wait=True)
        print(self.env.render())
        print(f"State: {state}")
        time.sleep(0.01)
    

class BFSAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
    
    def get_step_from_parent(self, node : BFS_node):
        node_coords = self.env.to_row_col(node.state)
        parent_coords = self.env.to_row_col(node.parent.state)
        steps = {
            (1, 0) : 0,
            (0, 1) : 1,
            (-1, 0) : 2,
            (0, -1) : 3
        }
        step_coords = (node_coords[0] - parent_coords[0], node_coords[1] - parent_coords[1])
        return steps.get(step_coords, -1)
        
    def solution(self, node : BFS_node) -> List[int]:
        self.env.reset()
        actions = []
        while node.parent != None:
            #insert to the begining of the queue
            actions.insert(0, self.get_step_from_parent(node))
            node = node.parent
        return actions
    
    def solution_cost(self, node : BFS_node) -> int:
        total_cost = 0
        while node.parent != None:
            total_cost += self.env.succ(node.parent.state)[self.get_step_from_parent(node)]
        return total_cost
    
    def solution_cost_2(self, actions : List[int]) -> int:
        self.env.reset()
        total_cost = 0
        for action in actions:
            a, cost, b = self.env.step(action)
            total_cost += cost
        return total_cost
    
    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        self.expanded = 0

        open = []
        close = []

        # initialize node
        node = BFS_node(env.get_initial_state())

        if self.env.is_final_state(node.state): # should check if terminated? only if initial state can be a hole
            return (self.solution(node), total_cost, self.expanded)
        
        # add node to open
        open.append(node)

        # start searching
        while len(open) != 0:
            # remove node from open then add it to close
            node = open.pop(0)
            # step to new position - using step so the dragon ball values of the current state changes correctly
            close.append(node.state)

            self.expanded += 1
            if node.terminated == True:
                continue
            # iterate over the children of node

            for action, (state, cost, terminated) in self.env.succ(node.state).items():
                # check if action is legal
                if state == None:
                    continue

                # should update current state before steping
                env.set_state_2(node.state)
                env.step(action)

                # add new node
                child = BFS_node(env.get_state(), node)

                # BFS-G condition
                if (child not in open) and child.state not in close:
                    # check if a child is a goal
                    if self.env.is_final_state(child.state):
                        path = self.solution(child)
                        return (path , self.solution_cost_2(path), self.expanded)
                    # check if the current child is a hole
                    if terminated == True:
                        node.terminated = True
                    # add child to open queue
                    open.append(child)

            # draw the current situation (where is the agent now)
            self.animation(node.state)
        return None # will never reach this because there is always a goal state


class WeightedAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError
    

if __name__ == '__main__':
    MAPS = {
            "4x4": ["SFFF",
                    "FDFF",
                    "FFFD",
                    "FFFG"],
            "8x8": [
                "SFFFFFFF",
                "FFFFFTAL",
                "TFFHFFTF",
                "FFFFFHTF",
                "FAFHFFFF",
                "FHHFFFHF",
                "DFTFHDTL",
                "FLFHFFFG",
            ],
        }
    env = DragonBallEnv(MAPS["8x8"])
    state = env.reset()
    BFS_agent = BFSAgent()
    actions, total_cost, expanded = BFS_agent.search(env)
    print(f"Total_cost: {total_cost}")
    print(f"Expanded: {expanded}")
    print(f"Actions: {actions}")

    assert total_cost == 119.0, "Error in total cost returned"