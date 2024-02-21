from graphviz import Digraph

def generate_state_graph(map):
    rows = len(map)
    cols = len(map[0])

    dot = Digraph()
    dot.attr(rankdir='LR', size='19.2,10.8')  # Set size to 1920x1080 (1080p)

    # Create nodes
    for row in range(rows):
        for col in range(cols):
            state_id = f"{row}_{col}"
            label = f"({row+1}, {col+1})\\n{map[row][col]}"
            dot.node(state_id, label=label, fontsize="14", width="1", height="1")

    # Create edges
    for row in range(rows):
        for col in range(cols):
            state_id = f"{row}_{col}"
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < rows and 0 <= new_col < cols:
                    new_state_id = f"{new_row}_{new_col}"
                    dot.edge(state_id, new_state_id, penwidth="0.5")

    dot.render('state_graph', format='png', cleanup=True)

# Example map
map = [
    "SFFF",
    "FDFF",
    "FFFD",
    "FFFG"
]

# Example usage
generate_state_graph(map)

print("State graph generated.")

