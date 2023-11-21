from queue import LifoQueue

class State:
    def __init__(self, missionaries, cannibals, boat):
        self.missionaries = missionaries
        self.cannibals = cannibals
        self.boat = boat

    def is_valid(self):
        if self.missionaries < 0 or self.cannibals < 0 or self.missionaries > 3 or self.cannibals > 3:
            return False
        if self.missionaries < self.cannibals and self.missionaries > 0:
            return False
        if 3 - self.missionaries < 3 - self.cannibals and 3 - self.missionaries > 0:
            return False
        return True

    def is_goal(self):
        return self.missionaries == 0 and self.cannibals == 0 and self.boat == 0

    def __eq__(self, other):
        return (
            self.missionaries == other.missionaries and
            self.cannibals == other.cannibals and
            self.boat == other.boat
        )

    def __hash__(self):
        return hash((self.missionaries, self.cannibals, self.boat))

    def __repr__(self):
        return f"State({self.missionaries}, {self.cannibals}, {self.boat})"


def get_successors(current_state):
    successors = []
    moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]

    for m, c in moves:
        if current_state.boat == 1:
            successor = State(
                current_state.missionaries - m,
                current_state.cannibals - c,
                0
            )
        else:
            successor = State(
                current_state.missionaries + m,
                current_state.cannibals + c,
                1
            )

        if successor.is_valid() and successor != current_state:
            successors.append(successor)

    return successors


def dfs():
    start_state = State(3, 3, 1)
    goal_state = State(0, 0, 0)

    visited = set()
    stack = LifoQueue()
    stack.put((start_state, []))

    while not stack.empty():
        current_state, path = stack.get()

        if current_state.is_goal():
            return path

        visited.add(current_state)

        successors = get_successors(current_state)
        for successor in successors:
            if successor not in visited:
                stack.put((successor, path + [successor]))

    return None

def print_solution(solution):
    if solution is None:
        print("No solution found.")
    else:
        for i, state in enumerate(solution):
            print(f"Step {i + 1}: {state}")

if __name__ == "__main__":
    solution = dfs()
    print_solution(solution)
