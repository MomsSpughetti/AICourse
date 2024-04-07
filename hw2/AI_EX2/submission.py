from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


# TODO: section a : 3
def aRobotHasThePackage(env: WarehouseEnv, package):
    for r in env.robots:
        if r.package == package:
            return True
    return False

def manhattanDist(a, b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def canReachAndDeliverPackage(env: WarehouseEnv, robot, package):
    totalDistToDeliver = manhattanDist(robot.position, package.position) + manhattanDist(package.destination, package.position)
    return 1 if (robot.battery - totalDistToDeliver) >= 0 else 0

def canReachPosition(env: WarehouseEnv, robot, pos):
    totalDistToDeliver = manhattanDist(robot.position, pos)
    return 1 if (robot.battery - totalDistToDeliver) >= 0 else 0

def mostWorthyPackage(env: WarehouseEnv, robot):
    """
    returns the best package to deliver from the ones available to pick up and can be delivered
    """
    bestPackage = None
    bestPackageEvaluation = -25
    for p in env.packages[0:2]:
        currEval = -manhattanDist(robot.position, p.position) + manhattanDist(p.destination, p.position)
        if currEval > bestPackageEvaluation and not aRobotHasThePackage(env, p) and canReachAndDeliverPackage(env, robot, p):
            bestPackage = p
            bestPackageEvaluation = currEval
    return bestPackage

def mostWorthyPackageThatIsReachable(env: WarehouseEnv, robot):
    """
    returns the best package to deliver from the ones available to pick up and are reachable (not necessarily deliverable)
    """
    bestPackage = env.packages[0]
    bestPackageEvaluation = -25
    for p in env.packages[0:2]:
        currEval = -manhattanDist(robot.position, p.position) + manhattanDist(p.destination, p.position)
        if currEval < bestPackageEvaluation and not aRobotHasThePackage(env, p) and canReachPosition(env, robot, p.position):
            bestPackage = p
            bestPackageEvaluation = currEval
    return bestPackage

def closestChargingStation(env: WarehouseEnv, robot):
    csList = env.charge_stations
    cs1D = manhattan_distance(csList[0].position, robot.position)
    cs2D = manhattan_distance(csList[1].position, robot.position)
    return csList[0] if cs1D > cs2D else csList[1]

        
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    # helpers
    robot = env.get_robot(robot_id)
    MWP = mostWorthyPackage(env, robot)
    helper = robot.credit/3 if robot.credit > 0 else 1

    # !R.P
    if robot.package is None and MWP is not None:
        return (robot.credit) *(helper) + (robot.battery - manhattan_distance(MWP.position, MWP.destination)-manhattan_distance(robot.position, MWP.position) )
    
    #helpers
    CCS = closestChargingStation(env, robot)
    canReachChargeStation = canReachPosition(env, robot, CCS.position)

    # !R.P = None and battery is not enough
    # in case can not charge => will try to pick a package to interrupt the other robot.
    MWPToInterrupt = mostWorthyPackageThatIsReachable(env, robot)
    if robot.package is None and MWP is None: 
        return (canReachChargeStation) * (robot.battery - manhattan_distance(CCS.position, robot.position)) \
            + (1 - canReachChargeStation) * (-manhattan_distance(robot.position, MWPToInterrupt.position))
        # return robot.battery - manhattan_distance(CCS.position, robot.position)
    # R.p
    # will deliver (guaranteed based on previous conditions)
    if robot.package is not None:
        return (robot.credit)*(helper) \
            + (robot.battery -manhattan_distance(robot.package.destination, robot.position))
    
    # will not reach here
    return robot.credit

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        # 
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
#        self.trajectory = ["move east", "pick_up", "move east", "move east", "move west", "move west", "move west", "move west", "move south", "move south", "drop_off"]
        self.trajectory = [
            "move east",
            "move north",
            "pick up",
            "move north",
            "move west",
            "drop off"

        ]
        # "move north", "pick_up", "move east", "move north", "move north", "pick_up", "move east", "move east",
        # "move south", "move south", "move south", "move south", "drop_off"

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)