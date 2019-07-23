
import loader

import sys
sys.path.append('/home/barbara/scheduling_procedures/TAEMS')
sys.path.append('/home/barbara/scheduling_procedures/Genetic algorithm')
sys.path.append('/home/barbara/scheduling_procedures/Tabu search')
sys.path.append('/home/barbara/scheduling_procedures/Simulated annealing')
sys.path.append('/home/barbara/scheduling_procedures/Ant colony optimization')

from taems import TaemsTree
import genetic_algorithm
import tabu_search
import simulated_annealing
import ant_colony_optimization

if __name__ == "__main__":
    [alternative, pt, rt, dt] = loader.load_file("./Input/", "alternative_1.txt")
    tree = TaemsTree.load_from_file("./Input/example_large.taems")

    alg = 0
    if alg == 0:
        ga = genetic_algorithm.GeneticAlgorithm(200, 0.1, 0.1)
        ga.optimize(tree, alternative, pt, rt, dt, 20)
    elif alg == 1:
        tabu = tabu_search.TabuSearch(10)
        tabu.optimize(tree, alternative, pt, rt, dt, 100)
    elif alg == 2:
        sa = simulated_annealing.SimulatedAnnealing(1000, 0.99)
        sa.optimize(tree, alternative, pt, rt, dt, 100)
    elif alg == 3:
        aco = ant_colony_optimization.AntColonyOptimization(10, 0.0, 0.2)
        aco.optimize(tree, alternative, pt, rt, dt, 5)
