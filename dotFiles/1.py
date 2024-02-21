def generate_dot(rows, cols, map):
    dot_code = "digraph state_graph {\n"
    dot_code += "    node [shape=circle]\n"
    
    # Create nodes
    for row in range(rows):
        for col in range(cols):
            for db1 in [True, False]:
                for db2 in [True, False]:
                    state_id = f"{row}{col}{int(db1)}{int(db2)}"
                    label = f"({row+1}, {col+1})\\nDB1Collected: {db1}\\nDB2Collected: {db2}"
                    dot_code += f'    {state_id} [label="{label}"]\n'
    
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
                                    dot_code += f"    {state_id_1} -> {state_id_2} [label=\"\", color=\"blue\"];\n"

    dot_code += "}\n"
    return dot_code

# Example map
map = [
    "SF",
    "FD"
]

rows = len(map)
cols = len(map[0])

# Example usage
dot_code = generate_dot(rows, cols, map)

# Write DOT code to a file
filename = "state_graph.dot"
with open(filename, "w") as file:
    file.write(dot_code)

print(f"DOT code written to {filename}")

# Generate the graph using the

