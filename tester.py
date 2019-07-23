
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

    test = 2
    if test == 0:
        ga_solutions = []

        ga = genetic_algorithm.GeneticAlgorithm(200, 0.1, 0.3)
        for i in range(1000):
            [best_sol, iteration] = ga.optimize(tree, alternative, pt, rt, dt, 50)
            print([i, best_sol.total_tardiness, iteration])
            ga_solutions.append([best_sol, iteration])

        f = open("Results/ga_results.txt", "w")
        for i in range(1000):
            f.write(str(ga_solutions[i][0].total_tardiness) + " " + str(ga_solutions[i][1]) + "\n")
        f.close()
    elif test == 1:
        tabu_solutions = []

        tabu = tabu_search.TabuSearch(10)
        for i in range(1000):
            [best_sol, iteration] = tabu.optimize(tree, alternative, pt, rt, dt, 100)
            print([i, best_sol.total_tardiness, iteration])
            tabu_solutions.append([best_sol, iteration])

        f = open("Results/tabu_results_2.txt", "w")
        for i in range(1000):
            f.write(str(tabu_solutions[i][0].total_tardiness) + " " + str(tabu_solutions[i][1]) + "\n")
        f.close()
    elif test == 2:
        sa_solutions = []

        sa = simulated_annealing.SimulatedAnnealing(1000, 0.99)
        for i in range(1000):
            [best_sol, iteration] = sa.optimize(tree, alternative, pt, rt, dt, 100)
            print([i, best_sol.total_tardiness, iteration])
            sa_solutions.append([best_sol, iteration])

        f = open("Results/sa_results_4.txt", "w")
        for i in range(1000):
            f.write(str(sa_solutions[i][0].total_tardiness) + " " + str(sa_solutions[i][1]) + "\n")
        f.close()
    elif test == 3:
        aco_solutions = []

        aco = ant_colony_optimization.AntColonyOptimization(10, 0.0, 0.2)
        for i in range(250):
            [best_sol, iteration] = aco.optimize(tree, alternative, pt, rt, dt, 5)
            print([i, best_sol.total_tardiness, iteration])
            aco_solutions.append([best_sol, iteration])

        f = open("Results/aco_results.txt", "w")
        for i in range(250):
            f.write(str(aco_solutions[i][0].total_tardiness) + " " + str(aco_solutions[i][1]) + "\n")
        f.close()
