import random
from tqdm import tqdm  # tqdm is used to display a progress bar

class Puzzle:
    def __init__(self, n, target):
        """
        Puzzle class represents the puzzle with a specified size 'n' and a target arrangement.

        Parameters:
        - n (int): Size of the puzzle (n x n)
        - target (list): Target arrangement to be achieved in the puzzle.
        """
        self.n = n
        self.target = target

class PuzzleIndividual:
    def __init__(self, puzzle, arrangement=None):
        """
        PuzzleIndividual class represents an individual solution in the genetic algorithm.

        Parameters:
        - puzzle (Puzzle): The puzzle instance for which the individual is generated.
        - arrangement (list, optional): Initial arrangement of the puzzle. If not provided, a random arrangement is generated.
        """
        self.puzzle = puzzle
        self.arrangement = arrangement if arrangement else self.generate_arrangement()

    def generate_arrangement(self):
        """
        Generates a random arrangement for the puzzle.

        Returns:
        - list: A randomly generated arrangement for the puzzle.
        """
        arrangement = list(range(0, self.puzzle.n**2))
        random.shuffle(arrangement)
        return arrangement

    def fitness(self):
        """
        Calculates the fitness of the individual by counting the number of elements in the wrong position.

        Returns:
        - int: Fitness value representing the number of elements in the wrong position.
        """
        return sum(a != b for a, b in zip(self.arrangement, self.puzzle.target))

class GeneticAlgorithm:
    def __init__(self, crossover_rate, mutation_rate, population_size, max_generations):
        """
        GeneticAlgorithm class represents the main genetic algorithm for solving the puzzle.

        Parameters:
        - crossover_rate (float): Probability of crossover occurring during reproduction.
        - mutation_rate (float): Probability of mutation occurring in an individual.
        - population_size (int): Size of the population in each generation.
        - max_generations (int): Maximum number of generations for the algorithm.
        """
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.max_generations = max_generations

    def generate_initial_population(self, individual_class, puzzle, population_size=50):
        """
        Generates the initial population of individuals.

        Parameters:
        - individual_class (class): Class representing the individuals in the population.
        - puzzle (Puzzle): The puzzle instance for which the population is generated.
        - population_size (int): Size of the population to be generated.

        Returns:
        - list: Initial population of individuals.
        """
        unique_arrangements = set()
        population = []

        while len(population) < population_size:
            individual = individual_class(puzzle)
            arrangement_tuple = tuple(individual.arrangement)

            # Check if the arrangement is unique
            if arrangement_tuple not in unique_arrangements:
                unique_arrangements.add(arrangement_tuple)
                population.append(individual)

        return population

    def selection(self, population):
        """
        Selects individuals from the population based on their fitness.

        Parameters:
        - population (list): Current population of individuals.

        Returns:
        - list: Selected individuals based on their fitness.
        """
        parents = sorted(population, key=lambda ind: ind.fitness(), reverse=False)[:len(population)//2]
        return parents

    def crossover(self, parents):
        """
        Performs crossover between pairs of parents to produce offspring.

        Parameters:
        - parents (list): List of parent individuals.

        Returns:
        - tuple: Two new individuals representing the offspring after crossover.
        """
        point = random.randint(1, parents[0].puzzle.n**2 - 1)
        child1_arrangement = parents[0].arrangement[:point] + parents[1].arrangement[point:]
        child2_arrangement = parents[1].arrangement[:point] + parents[0].arrangement[point:]
        return PuzzleIndividual(parents[0].puzzle, child1_arrangement), PuzzleIndividual(parents[0].puzzle, child2_arrangement)

    def mutate(self, individual):
        """
        Performs mutation on an individual by swapping two randomly chosen elements.

        Parameters:
        - individual (PuzzleIndividual): Individual to undergo mutation.
        """
        pos1, pos2 = random.sample(range((individual.puzzle.n**2)-1), k=2)
        individual.arrangement[pos1], individual.arrangement[pos2] = individual.arrangement[pos2], individual.arrangement[pos1]

    def next_generation(self, population):
        """
        Generates the next generation of individuals through selection, crossover, and mutation.

        Parameters:
        - population (list): Current population of individuals.

        Returns:
        - list: Next generation of individuals.
        """
        parents = self.selection(population)
        next_gen = parents
        unique_arrangements = set()

        for i in range(0, len(parents), 2):
            if random.random() < self.crossover_rate:
                children = self.crossover([parents[i], parents[i+1]])
            else:
                children = [parents[i], parents[i+1]]

            for child in children:
                if random.random() < self.mutation_rate:
                    self.mutate(child)

                arrangement_tuple = tuple(child.arrangement)
                while arrangement_tuple in unique_arrangements:
                    self.mutate(child)
                    arrangement_tuple = tuple(child.arrangement)

                next_gen.append(child)
                unique_arrangements.add(arrangement_tuple)

        return next_gen[:len(population)]

    def solve_puzzle(self, puzzle):
        """
        Solves the puzzle using the genetic algorithm.

        Parameters:
        - puzzle (Puzzle): The puzzle to be solved.

        Returns:
        - PuzzleIndividual: Best solution found by the algorithm.
        """
        population = self.generate_initial_population(PuzzleIndividual, puzzle, self.population_size)

        population = sorted(population, key=lambda ind: ind.fitness(), reverse=False)
        best = population[0]

        for _ in tqdm(range(self.max_generations)):
            population = self.next_generation(population)
            population = sorted(population, key=lambda ind: ind.fitness(), reverse=False)
            best = population[0]
            if best.fitness() == 0:
                break

        return best

def display_arrangement(arrangement):
    """
    Displays the arrangement of the puzzle in a readable format.

    Parameters:
    - arrangement (list): The arrangement of the puzzle.

    Returns:
    - str: Formatted string representation of the puzzle arrangement.
    """
    display = "\n"
    for i in range(0, len(arrangement), n):
        display += " ".join((str(num) + " ") if num < 10 else str(num) for num in arrangement[i:i+n]) + "\n"
    return display

if __name__ == '__main__':
    # Parameters for the puzzle and genetic algorithm
    n = int(input("Enter the value of n (default is 4): ") or 3)
    target_arrangement = list(range(1, n**2))
    target_arrangement += [0]
    crossover_rate = float(
    input("Enter the crossover rate (default is 0.5): ") or 0.5)
    mutation_rate = float(
    input("Enter the mutation rate (default is 0.2): ") or 0.2)
    population_size = int(
    input("Enter the population size (default is 10000): ") or 10000)
    max_generations = int(
    input("Enter the maximum number of generations (default is 150): ") or 150)

    # Print puzzle information
    print("\nn :", n)
    print("target_arrangement :", display_arrangement(target_arrangement))

    # Create an instance of the GeneticAlgorithm class
    genetic_algorithm = GeneticAlgorithm(crossover_rate, mutation_rate, population_size, max_generations)

    # Create an instance of the Puzzle class
    puzzle = Puzzle(n, target_arrangement)

    # Solve the puzzle using the genetic algorithm
    best_solution = genetic_algorithm.solve_puzzle(puzzle)

    # Display the best solution and its fitness
    print(f"Best solution: {display_arrangement(best_solution.arrangement)}")
    print(f"Fitness: {best_solution.fitness()} ({best_solution.fitness()} wrong place out of {n**2})\n")
    print(f"{round((((n**2) - best_solution.fitness())/(n**2))*100, 2)}%")