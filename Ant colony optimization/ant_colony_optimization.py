#!/usr/bin/env python

__author__ = 'barbanas'

from copy import deepcopy
import random
import time
import numpy as np
from hashlib import md5
import simulator
from random import sample
import itertools
import sys
sys.path.append('/home/barbara/scheduling_procedures/Tabu search')
from tabu_search import TabuSearch


class AntColonyOptimization(object):
    """
    Ant colony optimization for scheduling unordered list of tasks.

    Attributes:

    """

    def __init__(self, ant_number, init_pheromone, evaporation):
        """
        Args:
        """
        self.best_solution = None
        self.current_solution = None
        self.initialDurationEV = {}
        self.pheromone = None
        self.ant_number = ant_number
        self.tasks = []
        self.init_pheromone = init_pheromone
        self.evaporation = evaporation

    def optimize(self, tree, alternative, task_duration, release_dates, due_dates, no_iter):
        """
        A method that starts ant colony optimization algorithm for generating schedule from unordered list of methods.

        :param tree: TaemsTree with hierarchical task structure
        :param alternative: an unordered list of actions to schedule
        :param task_duration: dictionary, key: action id, value: action duration
        :param release_dates: dictionary, key: action id, value: action due date
        :param due_dates: dictionary, key: action id, value: action release date
        :param no_iter: number of iterations of the algorithm
        :return: [(Solution) best found solution, (int) iteration at which the solution was found]
        """
        sim = simulator.LightSimulator(tree)
        t = deepcopy(alternative)
        # tasks.extend(nonLocalTasks)
        sim.execute_alternative(t, t)
        tasks_to_complete = sim.completedTasks

        start = time.time()

        tabu = TabuSearch(5)
        self.tasks = [x for x in alternative]
        self.pheromone = np.full((len(self.tasks), len(self.tasks)), self.init_pheromone)

        iteration = 0
        best_solution_iter = 0
        while iteration < no_iter:
            solutions = []
            ratings = []
            for i in range(self.ant_number):
                solution = self.generate_solution(tree, alternative, tasks_to_complete, task_duration, due_dates)
                solution.evaluate(due_dates, release_dates)

                [optimized, _] = tabu.optimize(tree, alternative, task_duration, release_dates, due_dates, 10, False)

                solutions.append(optimized)
                ratings.append(optimized.total_tardiness)

            if self.best_solution is None:
                self.best_solution = solutions[ratings.index(min(ratings))]
                best_solution_iter = iteration

            elif min(ratings) < self.best_solution.total_tardiness:
                self.best_solution = solutions[ratings.index(min(ratings))]
                best_solution_iter = iteration

            self.evaporate_pheromones()
            for item in solutions:
                self.add_pheromone(item)
            iteration += 1

        print(time.time() - start)
        print("best solution: " + str(self.best_solution.total_tardiness))
        self.best_solution.print_schedule()

        return [self.best_solution, best_solution_iter]

    def generate_solution(self, tree, alternative, tasks_to_complete, task_duration, due_dates):
        """
        A method that creates a solution to the scheduling problem.

        Args:
            alternative - a list of unordered methods
        """
        sim = simulator.LightSimulator(tree)
        sim.init_disablements(alternative, tasks_to_complete)
        hard_constrained = list(sim.hardConstrainedTasks)

        sol_order = []
        base = []
        i = 0
        while len(sol_order) < len(alternative):
            enabled_tasks = sim.get_enabled_tasks(alternative)
            to_schedule = list(enabled_tasks - set(sol_order))
            # choose one of tasks
            indices = []
            for task in to_schedule:
                indices.append(self.tasks.index(task))
            pheromones = [self.pheromone[i][j] for j in indices]
            fitness = []
            # output = ""
            for j in range(len(to_schedule)):
                if due_dates[to_schedule[j]] > 0:
                    fitness.append(pheromones[j] / pow(due_dates[to_schedule[j]], 2))
                    # output += '\033[91m' + str(fitness[-1]) + '\033[0m '
                    # print(due_dates[to_schedule[j]])
                else:
                    fitness.append(pheromones[j] / pow(500, 5))
                    # output += str(fitness[-1]) + ' '
            # print(output)
            if max(fitness) == 0:
                normalized_fitness = [1.0/len(fitness) for _ in range(len(fitness))]
            else:
                normalized_fitness = [x/sum(fitness) for x in fitness]

            sol_order.append(to_schedule[AntColonyOptimization.roulette_wheel(normalized_fitness)])
            sim.execute_task(sol_order[-1], tasks_to_complete)
            if sol_order[-1] in hard_constrained:
                base.append(i)
            i += 1

        solution = Solution(sol_order, base, task_duration)

        return solution

    @staticmethod
    def roulette_wheel(fitness):
        s = sum(fitness)
        prob_sum = 0

        win = random.random()
        for i in range(len(fitness)):
            prob_sum += fitness[i] / s
            if win <= prob_sum:
                return i

    def evaporate_pheromones(self):
        # old = self.pheromone[0][0]
        self.pheromone *= (1 - self.evaporation)
        # print(old - self.pheromone[0][0])

    def add_pheromone(self, solution):
        i = 0
        for item in solution.order:
            j = self.tasks.index(item)
            self.pheromone[i][j] += self.evaporation / solution.total_tardiness
            # print(1.0 / self.best_solution.total_tardiness)
            # input()
            i += 1


class Solution(object):
    """
    Attributes:
        genes - a list of methods in a schedule
        base - a list of positions of base methods
        rating - chromosome rating (for explanation see GeneticAlgorithm description)
        fitness - chromosome fitness (for explanation see GeneticAlgorithm description)
        durationEV - dictionary : { key = method label, value = expected value of duration}
        schedule - list [[method1 label, method1 start time, method1 end time], ...  ]
        ID - id of a chromosome - unique identifier that is calculated from order of methods in a genome
        scheduleDuration - expected duration of schedule
    """

    def __init__(self, order, base, method_duration):
        """
        Args:
             genes - an ordered list of methods
             base - a list of positions of base methods
             initialDurationEV - initial expected value for every methods duration (before they are scheduled)
        """
        self.order = deepcopy(order)
        self.base = deepcopy(base)
        self.total_tardiness = 0
        self.durationEV = deepcopy(method_duration)
        self.schedule = []
        self.ID = None
        self.scheduleDuration = 0
        self.tardy_jobs = {}

    def __str__(self):
        output = ""
        for i in range(len(self.order)):
            if i in self.base:
                output += '\033[92m' + str(self.order[i]) + '\033[0m '
            else:
                output += str(self.order[i]) + ' '
        return output

    def print_schedule(self):
        output = ""
        for item in self.schedule:
            if item[0] in self.tardy_jobs.keys():
                output += '\033[91m' + str(item) + '\033[0m '
            else:
                output += str(item) + ' '

        print(output)

    def evaluate(self, due_dates, release_dates):
        """
        A method that creates a schedule from ordered list of methods (self.genes). It also calculates
        rating of created schedule.

        Args:
            tree - taemsTree instance that holds the task structure
        """
        self.calc_id()

        self.schedule = []
        self.tardy_jobs = {}
        self.total_tardiness = 0

        start_time = [0]
        for method in self.order:

            if method in release_dates.keys():
                diff = release_dates[method] - start_time[-1]
                # insert slack
                if diff > 0:
                    start_time.append(start_time[-1] + diff)
                    self.schedule.append(["slack", start_time[-2], start_time[-1]])

            start_time.append(start_time[-1] + self.durationEV[method])
            self.schedule.append([method, start_time[-2], start_time[-1]])

            if due_dates[method] > 0:
                tardiness = self.schedule[-1][2] - due_dates[method]
                if tardiness > 0:
                    self.total_tardiness += tardiness
                    self.tardy_jobs[method] = tardiness

        self.scheduleDuration = self.schedule[-1][2]

    def calc_id(self):
        string = ""
        for x in self.order:
            string += x

        self.ID = md5(string).digest()

    def get_id(self):
        return self.ID

    def get_base(self):
        """
        A method that returns chromosome base methods - elements on self.base positions.
        """
        base_methods = []
        for x in self.base:
            base_methods.append(self.order[x])

        return base_methods
