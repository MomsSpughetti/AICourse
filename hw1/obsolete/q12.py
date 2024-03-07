from Algorithms_meged import *
import csv

if __name__ == "__main__":

    test_boards = {
    "map12x12": 
    ['SFAFTFFTHHHF',
    'AFLTFFFFTALF',
    'LHHLLHHLFTHD',
    'HALTHAHHADHF',
    'FFFTFHFFAHFL',
    'LLTHFFFAHFAT',
    'HAAFFALHTATF',
    'LLLFHFFHTLFH',
    'FATAFHTTFFAF',
    'HHFLHALLFTLF',
    'FFAFFTTAFAAL',
    'TAAFFFHAFHFG'],
    "map15x15": 
    ['SFTTFFHHHHLFATF',
    'ALHTLHFTLLFTHHF',
    'FTTFHHHAHHFAHTF',
    'LFHTFTALTAAFLLH',
    'FTFFAFLFFLFHTFF',
    'LTAFTHFLHTHHLLA',
    'TFFFAHHFFAHHHFF',
    'TTFFLFHAHFFTLFD',
    'TFHLHTFFHAAHFHF',
    'HHAATLHFFLFFHLH',
    'FLFHHAALLHLHHAT',
    'TLHFFLTHFTTFTTF',
    'AFLTDAFTLHFHFFF',
    'FFTFHFLTAFLHTLA',
    'HTFATLTFHLFHFAG'],
    "map20x20" : 
    ['SFFLHFHTALHLFATAHTHT',
    'HFTTLLAHFTAFAAHHTLFH',
    'HHTFFFHAFFFFAFFTHHHT',
    'TTAFHTFHTHHLAHHAALLF',
    'HLALHFFTHAHHAFFLFHTF',
    'AFTAFTFLFTTTFTLLTHDF',
    'LFHFFAAHFLHAHHFHFALA',
    'AFTFFLTFLFTAFFLTFAHH',
    'HTTLFTHLTFAFFLAFHFTF',
    'LLALFHFAHFAALHFTFHTF',
    'LFFFAAFLFFFFHFLFFAFH',
    'THHTTFAFLATFATFTHLLL',
    'HHHAFFFATLLALFAHTHLL',
    'HLFFFFHFFLAAFTFFDAFH',
    'HTLFTHFFLTHLHHLHFTFH',
    'AFTTLHLFFLHTFFAHLAFT',
    'HAATLHFFFHHHHAFFFHLH',
    'FHFLLLFHLFFLFTFFHAFL',
    'LHTFLTLTFATFAFAFHAAF',
    'FTFFFFFLFTHFTFLTLHFG']}

    test_envs = {}
    for board_name, board in test_boards.items():
        test_envs[board_name] = DragonBallEnv(board)


    BFS_agent = BFSAgent()
    WAStar_agent = WeightedAStarAgent()

    weights = [0.5, 0.7, 0.9]

    agents_search_function = [
        BFS_agent.search,
    ]

    header = ['map',  "BFS-G cost",  "BFS-G expanded",\
            'WA* (0.5) cost', 'WA* (0.5) expanded', 'WA* (0.7) cost', 'WA* (0.7) expanded', 'WA* (0.9) cost', 'WA* (0.9) expanded']

with open("results.csv", 'w') as f:
  writer = csv.writer(f)
  writer.writerow(header)
  for env_name, env in test_envs.items():
    data = [env_name]
    for agent in agents_search_function:
      _, total_cost, expanded = agent(env)
      data += [total_cost, expanded]
    for w in weights:
        _, total_cost, expanded = WAStar_agent.search(env, w)
        data += [total_cost, expanded]

    writer.writerow(data)