def is_safe(node, color, graph, colors):
    for neighbor in graph[node]:
        if colors[neighbor] == color:
            return False
    return True

def color_map_util(graph, num_colors, colors, node):
    if node == len(graph):
        return True

    for color in range(1, num_colors + 1):
        if is_safe(node, color, graph, colors):
            colors[node] = color
            if color_map_util(graph, num_colors, colors, node + 1):
                return True
            colors[node] = 0  # Backtrack

    return False

def color_map(graph, num_colors):
    colors = [0] * len(graph)

    if not color_map_util(graph, num_colors, colors, 0):
        print("Solution does not exist.")
        return None

    return colors

if __name__ == "__main__":
    # Example graph represented as an adjacency list
    graph = {
        0: [1, 2, 3],
        1: [0, 2],
        2: [0, 1, 3],
        3: [0, 2]
    }

    num_colors = 3

    result = color_map(graph, num_colors)

    if result:
        print("Map coloring:", result)
    else:
        print("No solution exists.")
