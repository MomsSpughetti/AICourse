from graphviz import Digraph

def generate_dot(rows, cols, map):
    dot = Digraph()
    dot.attr(rankdir='LR', size='8,5')

    # Create nodes
    for row in range(rows):
        for col in range(cols):
            for db1 in [True, False]:
                for db2 in [True, False]:
                    state_id = f"{row}{col}{int(db1)}{int(db2)}"
                    label = f"({row+1}, {col+1})\\nDB1Collected: {db1}\\nDB2Collected: {db2}"
                    dot.node(state_id, label=label)

    # Create edges
    for row1 in range(rows):
        for col1 in range(cols):
            for db1_1 in [True, False]:
                for db2_1 in [True, False]:
                    state_id_1 = f"{row1}{col1}{int(db1_1)}{int(db2_1)}"
                    for action in ['up', 'down', 'left', 'right']:
                        new_row = row1
                        new_col = col1
                        if action == 'up':
                            new_row -= 1
                        elif action == 'down':
                            new_row += 1
                        elif action == 'left':
                            new_col -= 1
                        elif action == 'right':
                            new_col += 1
                        
                        if 0 <= new_row < rows and 0 <= new_col < cols:
                            for db1_2 in [True, False]:
                                for db2_2 in [True, False]:
                                    state_id_2 = f"{new_row}{new_col}{int(db1_2)}{int(db2_2)}"
                                    dot.edge(state_id_1, state_id_2)

    dot.render('state_graph', format='png', cleanup=True)

# Example map
map = [
    "SFFF",
    "FDFF",
    "FFFD",
    "FFFG"
]

rows = len(map)
cols = len(map[0])

# Example usage
generate_dot(rows, cols, map)

print("State graph generated.")

