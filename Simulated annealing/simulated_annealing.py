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
from math import exp


class SimulatedAnnealing(object):
    """
    Simulated annealing for scheduling unordered list of tasks.

    Attributes:

    """

    def __init__(self, T0, alfa):
        """
        Args:
        """
        self.best_solution = None
        self.current_solution = None
        self.initialDurationEV = {}
        self.temperature = T0
        self.cooling_factor = alfa

    def optimize(self, tree, alternative, task_duration, release_dates, due_dates, no_iter):
        """
        A method that starts simulated annealing algorithm for generating schedule from unordered list of methods.

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

        # generate initial solution
        self.current_solution = SimulatedAnnealing.generate_solution(tree, alternative, tasks_to_complete, task_duration)
        self.current_solution.evaluate(due_dates, release_dates)
        self.best_solution = self.current_solution

        iteration = 0
        best_solution_iter = 0
        while iteration < no_iter:
            [neighborhood, total_tardiness] = self.generate_neighborhood(self.current_solution, sim, tasks_to_complete,
                                                                         due_dates, release_dates)

            if random.random() < 0.6:
                best_index = total_tardiness.index(min(total_tardiness))
            else:
                best_index = SimulatedAnnealing.roulette_wheel(total_tardiness)

            # update current solution
            if self.current_solution.total_tardiness >= neighborhood[best_index].total_tardiness:
                self.current_solution = neighborhood[best_index]
            else:
                if random.random() < exp(-float(neighborhood[best_index].total_tardiness -
                                                self.current_solution.total_tardiness) / self.temperature):
                    self.current_solution = neighborhood[best_index]

            # update best solution
            if self.current_solution.total_tardiness < self.best_solution.total_tardiness:
                self.best_solution = self.current_solution
                best_solution_iter = iteration

            # update temperature
            self.temperature *= self.cooling_factor
            iteration += 1

        print(time.time() - start)
        print("best solution: " + str(self.best_solution.total_tardiness))
        self.best_solution.print_schedule()

        return [self.best_solution, best_solution_iter]

    @staticmethod
    def generate_solution(tree, alternative, tasks_to_complete, task_duration):
        """
        A method that creates a solution to the scheduling problem.

        Args:
            alternative - a list of unordered methods
        """
        sim = simulator.LightSimulator(tree)

        base, _, _ = SimulatedAnnealing.create_feasible_base_order(alternative, [], tasks_to_complete, sim)
        sim.setup()

        [methods, base_index] = SimulatedAnnealing.create_solution_from_base(base, alternative)

        solution = Solution(methods, base_index, task_duration)

        return solution

    @staticmethod
    def create_feasible_base_order(alternative, non_local_tasks, tasks_to_complete, sim):
        """
        Create a feasible order of hard-constrained methods in alternative (base of schedule).
        Args:
            alternative (list[str]): Labels of unordered methods (from taems task structure).
            non_local_tasks (list[str]): Labels of non local tasks (from taems task structure).
            tasks_to_complete
            sim (simulator.LightSimulator): Simulator used to simulate effects of activation of hard constraints.
        """
        to_complete = list(set(alternative) | set(non_local_tasks))
        sim.init_disablements(to_complete, tasks_to_complete)
        # sim.execute_alternative(nonLocalTasks, tasksToComplete)

        hard_constrained = list(sim.hardConstrainedTasks)

        # No tasks are hard constrained so base is not needed.
        if len(hard_constrained) == 0:
            return [[], False, False]

        temp = deepcopy(hard_constrained)
        feasible_base_order = []

        number_of_options = 0
        while True:
            possible_next = sim.get_enabled_tasks(temp)
            if len(set(possible_next) & set(to_complete)) == 0:
                break
            non_local_next = set(non_local_tasks) & set(possible_next)
            possible_next = set(alternative) & set(possible_next)

            sim.execute_alternative(non_local_next, tasks_to_complete)
            temp = list(set(temp) - set(non_local_next))
            number_of_options += len(possible_next)

            if len(possible_next) > 0:
                feasible_base_order.append(random.sample(possible_next, 1)[0])
                temp.remove(feasible_base_order[-1])
                if len(temp) == 0:
                    break
                sim.execute_task(feasible_base_order[-1], tasks_to_complete)

        if number_of_options == len(alternative):
            return [feasible_base_order, True, True]
        if number_of_options == len(hard_constrained):
            return [feasible_base_order, True, False]
        elif number_of_options == 0:
            return [[], True, False]
        else:
            return [feasible_base_order, False, False]

    @staticmethod
    def create_solution_from_base(base, alternative):
        """
        A method that creates solution from ordered list of base methods. First it selects indices for base
        methods and puts them into selected positions. Then all other methods are permuted in random order and
        put in available places in the schedule.

        Args:
            base - list of ordered base methods
            alternaive - unordered list of methods to schedule
        """
        schedule = ["" for _ in range(len(alternative))]

        # 1. position base
        base_indices = sample(range(len(alternative)), len(base))
        base_indices.sort()

        x = 0
        for i in base_indices:
            schedule[i] = base[x]
            x += 1

        if len(schedule) == len(base):
            return [schedule, base_indices]

        # 2. put other tasks into schedule
        permutation = list(np.random.permutation(list(set(alternative) - set(base))))

        for i in range(len(schedule)):
            if schedule[i] == "":
                schedule[i] = permutation.pop(0)
                if len(permutation) == 0:
                    break

        return [schedule, base_indices]

    @staticmethod
    def generate_neighborhood(solution, sim, to_complete, due_dates, release_dates):
        '''
        :param solution:
        :param sim:
        :param to_complete:
        :param due_dates:
        :param release_dates:
        :return:
        '''
        neighborhood = []
        total_tardiness = []
        neighborhood_move = itertools.combinations(range(len(solution.order)), 2)
        for move in neighborhood_move:
            order = deepcopy(solution.order)
            temp = order[move[0]]
            order[move[0]] = order[move[1]]
            order[move[1]] = temp
            sim.setup()
            sim.init_disablements(order, to_complete)
            error = sim.execute_alternative(order, to_complete)
            if not error:
                neighborhood.append(Solution(order, solution.base, solution.durationEV))
                neighborhood[-1].evaluate(due_dates, release_dates)
                total_tardiness.append(neighborhood[-1].total_tardiness)

        return [neighborhood, total_tardiness]

    @staticmethod
    def roulette_wheel(ratings):
        m = min(ratings)
        fitness = [m / float(x) for x in ratings]
        s = sum(fitness)
        prob_sum = 0

        win = random.random()
        for i in range(len(ratings)):
            prob_sum += fitness[i] / s
            if win <= prob_sum:
                return i


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

            '''if tree.tasks[method].deadline is not None:
                deadlines[0] += 1
                if start_time[-1] + tree.tasks[method].DurationEV - tree.tasks[method].deadline > 0:
                    deadlines[1] += 1

            if tree.tasks[method].earliestStartTime is not None:
                earliestStartTimes[0] += 1
                if tree.tasks[method].earliestStartTime - start_time[-1] > 0:
                    earliestStartTimes[1] += 1'''

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
