
from Algorithms_meged import *

if __name__ == '__main__':
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
    aS = WeightedAStarAgent()
    actions, total_cost, expanded = aS.search(env,0.5)
    print(f"Total_cost: {total_cost}")
    print(f"Expanded: {expanded}")
    print(f"Actions: {actions}")

    #assert total_cost == 119.0, "Error in total cost returned"