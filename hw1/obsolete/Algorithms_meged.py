import numpy as np
from time import sleep
from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
from DragonBallEnv import *
from IPython.display import clear_output
import heapdict


class Node:
    def __init__(self, state: Tuple, action, parent=None,  gVal: float = 0, fVal: float = 0, terminate: bool = False) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.g = gVal
        self.f = fVal
        self.terminate = terminate

    def toHeapKey(self):
        return self.state

    def toHeapValue(self):
        return (self.f, self.state, self)

    def getPath(self, env: DragonBallEnv) -> Tuple[List[int], float]:
        cost = 0
        actions = []
        currentNode = self
        while currentNode != None:
            actions.append(currentNode.action)
            if (currentNode.parent != None):
                cost += env.succ(currentNode.parent.state)[
                    currentNode.action][1]
            currentNode = currentNode.parent
        actions.pop()  # since the first one is not counted
        n = len(actions)
        # for some reason, reverse() won't work
        actions = [actions[n-i] for i in range(1, n+1)]
        return (actions, cost)


class BFSAgent():
    def __init__(self) -> None:
        pass

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        env.reset()
        expandedNodes = 0

        open = []
        openDict = dict()  # for faster checking
        closeDict = dict()

        # initialize node
        currentNode = Node(env.get_initial_state(), 0, None, terminate=False)

        # should check if terminated? only if initial state can be a hole
        if env.is_final_state(currentNode.state):
            path = currentNode.getPath(env)
            return (path[0], 0, expandedNodes)

        # add node to open
        open.append(currentNode)
        openDict[currentNode.state] = True

        # start searching
        while len(open) != 0:
            # remove node from open then add it to close
            currentNode = open.pop(0)
            del openDict[currentNode.state]
            # step to new position - using step so the dragon ball values of the current state changes correctly
            closeDict[currentNode.state] = True

            expandedNodes += 1
            if currentNode.terminate == True:
                continue
            # iterate over the children of node

            for action, (state, cost, terminated) in env.succ(currentNode.state).items():
                # check if action is legal
                if state == None:
                    continue

                # should update current state before steping
                env.set_state_2(currentNode.state)
                env.step(action)

                # add new node
                childNode = Node(env.get_state(), action,
                                 currentNode, terminate=terminated)

                # BFS-G condition
                if (childNode.state not in openDict) and (childNode.state not in closeDict):
                    # check if a child is a goal
                    if env.is_final_state(childNode.state):
                        path = childNode.getPath(env)
                        return (path[0], path[1], expandedNodes)

                    # add child to open queue
                    open.append(childNode)
                    openDict[childNode.state] = True

            # draw the current situation (where is the agent now)
        return None  # will never reach this because there is always a goal state


class WeightedAStarAgent():
    def __init__(self) -> None:
        pass

    @staticmethod
    def msap(state: Tuple, env: DragonBallEnv) -> float:
        dragonBalls = [env.d1, env.d2]
        targetList = [dragonBalls[i] for i in [0, 1]
                      if state[i+1] == False] + env.get_goal_states()
        coordList = [env.to_row_col(s) for s in targetList if s is not None]
        def manhattanDistance(v1, v2): return abs(
            v1[0] - v2[0]) + abs(v1[1] - v2[1])
        statePoint = env.to_row_col(state)
        return np.min([manhattanDistance(statePoint, s) for s in coordList])

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        # to not write env every time
        def h(state): return self.msap(state, env)
        env.reset()
        expandedNodes = 0
        startNode = Node(env.get_initial_state(), 0, parent=None, gVal=0,
                         fVal=h_weight*h(env.get_initial_state()), terminate=False)
        openHeap = heapdict.heapdict()
        closedHeap = heapdict.heapdict()
        openHeap[startNode.toHeapKey()] = startNode.toHeapValue()
        while len(openHeap) != 0:
            # Getting the node, the values is already in it.
            currentNode = openHeap.popitem()[1][2]
            closedHeap[currentNode.toHeapKey()] = currentNode.toHeapValue()

            if env.is_final_state(currentNode.state):
                path = currentNode.getPath(env)
                return (path[0], path[1], expandedNodes)

            expandedNodes += 1
            # in G but not enough dragon balls, or we've countered at a hole.
            if currentNode.terminate:
                continue

            for action, (state, cost, terminate) in env.succ(currentNode.state).items():
                if state is None:
                    continue

                env.set_state_2(currentNode.state)
                state, _, _ = env.step(action)
                newG = cost + currentNode.g
                newF = (1-h_weight)*newG + (h_weight)*h(state)

                inOpen = openHeap.get(state)
                inClose = closedHeap.get(state)

                if inOpen is None and inClose is None:
                    newNode = Node(state, action, parent=currentNode,
                                   gVal=newG, fVal=newF, terminate=terminate)
                    openHeap[newNode.toHeapKey()] = newNode.toHeapValue()

                elif inOpen is not None:
                    openNode = inOpen[2]
                    if (newF < openNode.f):
                        openNode.f = newF
                        openNode.g = newG
                        openNode.action = action
                        openNode.parent = currentNode
                        openHeap[openNode.toHeapKey()] = openNode.toHeapValue()

                else:
                    closeNode = inClose[2]
                    if (newF < closeNode.f):
                        closeNode.g = newG
                        closeNode.f = newF
                        closeNode.action = action
                        closeNode.parent = currentNode
                        openHeap[closeNode.toHeapKey(
                        )] = closeNode.toHeapValue()
                        closedHeap.pop(closeNode.state)


class AStarEpsilonAgent():
    def __init__(self) -> None:
        pass

    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        env.reset()
        expandedNodes = 0

        startNode = Node(env.get_initial_state(), 0,
                         parent=None, fVal=0, terminate=False)
        openHeap = heapdict.heapdict()
        closedHeap = heapdict.heapdict()
        focalHeap = heapdict.heapdict()

        openHeap[startNode.toHeapKey()] = startNode.toHeapValue()
        focalHeap[startNode.toHeapKey()] = startNode.toHeapValue()
        while len(focalHeap) != 0:
            currentNode = focalHeap.popitem()[1][2]
            openHeap.pop(currentNode.state)
            closedHeap[currentNode.toHeapKey()] = currentNode.toHeapValue()

            if env.is_final_state(currentNode.state):
                path = currentNode.getPath(env)
                return (path[0], path[1], expandedNodes)

            expandedNodes += 1
            # in G but not enough dragon balls, or we've countered at a hole.
            if currentNode.terminate:
                continue

            for action, (state, cost, terminate) in env.succ(currentNode.state).items():
                if state is None:
                    continue

                env.set_state_2(currentNode.state)
                state, _, _ = env.step(action)
                newF = cost + currentNode.f

                inOpen = openHeap.get(state)
                inClose = closedHeap.get(state)
                currentMinNode = focalHeap.peekitem()[1][2] if len(
                    focalHeap.keys()) > 0 else None

                if inOpen is None and inClose is None:
                    newNode = Node(state, action, parent=currentNode,
                                   fVal=newF, terminate=terminate)
                    openHeap[newNode.toHeapKey()] = newNode.toHeapValue()
                    if (currentMinNode is None or newF <= (1+epsilon)*currentMinNode.f):
                        focalHeap[newNode.toHeapKey()] = newNode.toHeapValue()

                elif inOpen is not None:
                    openNode = inOpen[2]
                    if (newF < openNode.f):
                        openNode.f = newF
                        openNode.action = action
                        openNode.parent = currentNode
                        openHeap[openNode.toHeapKey()] = openNode.toHeapValue()
                        if (newF < currentMinNode.f):
                            focalHeap[openNode.toHeapKey(
                            )] = openNode.toHeapValue()

                else:
                    closeNode = inClose[2]
                    if (newF < closeNode.f):
                        closeNode.f = newF
                        closeNode.action = action
                        closeNode.parent = currentNode
                        openHeap[closeNode.toHeapKey(
                        )] = closeNode.toHeapValue()
                        if (newF < currentMinNode.f):
                            focalHeap[closeNode.toHeapKey(
                            )] = closeNode.toHeapValue()
                        closeNode.pop(state)