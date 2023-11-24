import random

class VacuumCleaner:
    def __init__(self, location, environment_size):
        self.location = location
        self.environment_size = environment_size
        self.environment = [random.choice([0, 1]) for _ in range(environment_size)]

    def sense(self):
        return self.environment[self.location]

    def clean(self):
        self.environment[self.location] = 0
        print(f"Cleaned dirt at location {self.location}.")

    def move_left(self):
        if self.location > 0:
            self.location -= 1
        print(f"Moved left to location {self.location}.")

    def move_right(self):
        if self.location < self.environment_size - 1:
            self.location += 1
        print(f"Moved right to location {self.location}.")

def run_vacuum_cleaner(agent, steps):
    for _ in range(steps):
        dirt_status = agent.sense()
        if dirt_status == 1:
            agent.clean()
        if random.choice([True, False]):
            agent.move_left()
        else:
            agent.move_right()

if __name__ == "__main__":
    environment_size = 5  # Adjust the environment size as needed
    initial_location = random.randint(0, environment_size - 1)

    vacuum_agent = VacuumCleaner(initial_location, environment_size)
    print(f"Initial Environment: {vacuum_agent.environment}")

    steps_to_run = 10  # Adjust the number of steps the agent should take
    run_vacuum_cleaner(vacuum_agent, steps_to_run)

    print(f"Final Environment: {vacuum_agent.environment}")
