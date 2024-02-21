
from Algorithms import *

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