#!/usr/bin/env python

__author__ = 'barbanas'


from copy import deepcopy
import random
import time
from math import floor, factorial
from hashlib import md5
import numpy as np
import simulator
from random import sample
import itertools


class GeneticAlgorithm(object):
    """
    Genetic algorithm for scheduling unordered list of tasks.
    
    Attributes:
        populationSize - integer
        population - a list of Chromosome instances
        populationID - a list of string values (ID of every chromosome in a population)
        rating - rating (based on percentage of temporal and precedence constraints met by every chromosome)
        fitness - quality of every chromosome that unites rating and total schedule length
        bestSolution - a Chromosome instance, best solution of a scheduling problem (with max fitness)
        elitePercentage - percentage of population that is automatically promoted to next generation
        worstPercentage - percentage of population that is removed from population before creating new population
        initialDurationEV - initial expected value for every methods duration (before they are scheduled)

    """

    def __init__(self, popSize, elite, worst):
        """
        Args:
            popSize - population size
            elite - percentage of population that is automatically promoted to next generation
            worst - percentage of population that is removed from population before creating new population
        """
        self.populationSize = popSize
        self.population = []
        self.populationID = []
        self.rating = []
        self.fitness = []
        self.best_solution = None
        self.elitePercentage = elite
        self.worstPercentage = worst
        self.initialDurationEV = {}
        
    def optimize(self, tree, alternative, task_duration, release_dates, due_dates, no_iter):
        """
        A method that starts genetic algorithm for generating schedule from unordered list of methods.

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
        
        self.init_population(tree, alternative, tasks_to_complete, task_duration)

        self.evaluate_population(due_dates, release_dates)

        iteration = 0
        best_solution_iter = 0
        previous_best = min(self.rating)

        while iteration < no_iter:

            self.best_solution = self.population[self.rating.index(min(self.rating))]
            if self.best_solution.total_tardiness < previous_best:
                best_solution_iter = iteration
                previous_best = self.best_solution.total_tardiness

            next_gen = []
            next_gen_id = []
            next_gen_rating = []

            self.promote_elite(next_gen, next_gen_id, next_gen_rating)
            self.eliminate_the_worst()

            self.create_new_generation(next_gen, next_gen_id, next_gen_rating, due_dates, release_dates, 0, sim, tasks_to_complete)
            
            self.population = [x for x in next_gen]
            self.populationID = [x for x in next_gen_id]
            self.rating = [x for x in next_gen_rating]

            iteration += 1

        print(time.time() - start)
        print("best solution: " + str(self.best_solution.total_tardiness) + " " + str(best_solution_iter))
        self.best_solution.print_schedule()

        return [self.best_solution, best_solution_iter]

    def init_population(self, tree, alternative, tasks_to_complete, task_duration):
        """
        A method that creates initial population of feasible solutions to the scheduling problem.

        Args:
            alternative - a list of unordered methods
        """
        self.population = []

        self.initialDurationEV = {}
        for method in alternative:
            self.initialDurationEV[method] = task_duration[method]

        sim = simulator.LightSimulator(tree)

        while len(self.population) < self.populationSize:

            base, _, _ = self.create_feasible_base_order(alternative, [], tasks_to_complete, sim)
            sim.setup()

            [methods, base_index] = self.create_solution_from_base(base, alternative)

            chromosome = Chromosome(methods, base_index, self.initialDurationEV)
            chromosome.calc_id()

            if chromosome.ID in self.populationID:
                continue

            self.population.append(chromosome)
            self.populationID.append(chromosome.ID)

        return [None, False]
            
    def create_feasible_base_order(self, alternative, non_local_tasks, tasks_to_complete, sim):
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

    def create_solution_from_base(self, base, alternative):
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
        permutation = GeneticAlgorithm.permutation(list(set(alternative) - set(base)))

        for i in range(len(schedule)):
            if schedule[i] == "":
                schedule[i] = permutation.pop(0)
                if len(permutation) == 0:
                    break
        
        return [schedule, base_indices]

    @staticmethod
    def permutation(array):
        """
        Creates random permutation from a array.

        Args:
            array - a list of elements to premute
        """
        perm = []
        temp = deepcopy(array)

        while len(temp) > 0:
            perm.append(temp.pop(random.randint(0, len(temp)-1)))
            
        return perm

    def evaluate_population(self, due_dates, release_dates):
        """
        A method that evaluates population by calculating its rating and fitness.

        Args:

        """
        self.populationID = []
        self.rating = []
        
        for chromosome in self.population:
            chromosome.evaluate(due_dates, release_dates)
            self.rating.append(chromosome.total_tardiness)
            self.populationID.append(chromosome.get_id())

    def promote_elite(self, next_gen, next_gen_id, next_gen_rating):
        """
        A method that puts elitePercentage of best chromosomes in population into next generation.

        Args:
            next_gen (out) - list of chromosomes in next generation
            next_gen_id (out) - list of next generation chromosome IDs
            next_gen_rating (out) - list of next generation chromosome ratings
            
            ***(out) arguments are empty lists and are partially populated in this method
        """
        elite_number = int(floor(self.populationSize * self.elitePercentage))
        temp = deepcopy(self.rating)

        while elite_number > 0:
            x = temp.index(min(temp))
            temp[x] = -1

            next_gen.append(self.population[x])
            next_gen_id.append(self.populationID[x])
            next_gen_rating.append(self.rating[x])

            elite_number -= 1

    def eliminate_the_worst(self):
        """
        A method that removes worstPercentage of chromosomes with lowest fitness from current population. 
        They do not participate in transferring their genes into next generation.
        """
        worst_number = int(floor(self.populationSize * self.worstPercentage))
        
        removed = 0
        while removed < worst_number:
            x = self.rating.index(max(self.rating))

            self.rating.pop(x)
            self.population.pop(x)
            self.populationID.pop(x)

            removed += 1

    def roulette_wheel(self):
        m = min(self.rating)
        fitness = [m / float(x) for x in self.rating]
        s = sum(fitness)
        prob_sum = 0

        win = random.random()
        for i in range(len(self.population)):
            prob_sum += fitness[i] / s
            if win <= prob_sum:
                return i

    def create_new_generation(self, next_gen, next_gen_id, next_gen_rating, due_dates, release_dates, start_time, sim, to_complete):
        """
        A method that populates next generation with chromosomes created by applying genetic operators
        to parents from current population.
        
        Args:
            next_gen (out) - list of chromosomes in next generation
            next_gen_id (out) - list of next generation chromosome IDs
            next_gen_rating (out) - list of next generation chromosome ratings
            tree - taemsTree instance that holds the task structure
        """
        while len(next_gen) < self.populationSize:

            parents = [self.roulette_wheel(), self.roulette_wheel()]
            while parents[1] == parents[0]:
                parents[1] = self.roulette_wheel()

            if random.random() < 0.5:
                children = self.crossover_position_based(parents)
            else:
                children = self.crossover_order_based(parents)

            for child in children:
                # mutation
                if random.random() < 0.5:
                    '''if random.random() < 0.5:
                        m = self.mutation_order_based(child)
                    else:
                        m = self.mutation_position_based(child)
                    if len(m) > 0:
                        child = m[0]'''
                child = self.mutation(child, sim, to_complete)[0]
                child.evaluate(due_dates, release_dates)
                next_gen.append(child)
                next_gen_id.append(child.ID)
                next_gen_rating.append(child.total_tardiness)

    def crossover_position_based(self, parents):
        """
        Position based crossover. Children get order of (all) methods from one parent and base method
        positions from other parent. It only changes the position of base methods.

        Args:
            parents - two Chromosome instances
            tree - taemsTree instance that holds the task structure
        """
        p1 = self.population[parents[0]]
        p2 = self.population[parents[1]]

        child1 = Chromosome(["" for _ in range(len(p1.genes))], p2.base, self.initialDurationEV)
        child2 = Chromosome(["" for _ in range(len(p2.genes))], p1.base, self.initialDurationEV)

        # set up base
        for i in range(len(child1.base)):
            child2.genes[child2.base[i]] = p2.genes[p2.base[i]]
            child1.genes[child1.base[i]] = p1.genes[p1.base[i]]

        i1 = 0
        i2 = 0
        for i in range(len(p1.genes)):
            if len(child2.genes[i]) == 0:
                while i2 in p2.base:
                    i2 += 1
                child2.genes[i] = p2.genes[i2]
                i2 += 1

            if len(child1.genes[i]) == 0:
                while i1 in p1.base:
                    i1 += 1
                child1.genes[i] = p1.genes[i1]
                i1 += 1
         
        child1.calc_id()
        child2.calc_id()

        result = []
        if child1.ID not in self.populationID:
            result.append(child1)

        if child2.ID not in self.populationID:
            result.append(child2)

        return result

    def crossover_order_based(self, parents):
        """
        Order based crossover. Children get base method positions from one parent and base method order
        form other. It doesn't change the position of base methods, the only change is order of base.

        Args:
            parents - two Chromosome instances
            tree - taemsTree instance that holds the task structure
        """
        p1 = self.population[parents[0]]
        p2 = self.population[parents[1]]

        child1 = Chromosome(["" for _ in range(len(p1.genes))], p1.base, self.initialDurationEV)
        child2 = Chromosome(["" for _ in range(len(p2.genes))], p2.base, self.initialDurationEV)

        # set up base
        for i in range(len(child1.base)):
            child2.genes[child2.base[i]] = p1.genes[p1.base[i]]
            child1.genes[child1.base[i]] = p2.genes[p2.base[i]]

        i1 = 0
        i2 = 0
        for i in range(len(p1.genes)):
            if len(child2.genes[i]) == 0:
                while i1 in p1.base:
                    i1 += 1
                child2.genes[i] = p1.genes[i1]
                i1 += 1

            if len(child1.genes[i]) == 0:
                while i2 in p2.base:
                    i2 += 1
                child1.genes[i] = p2.genes[i2]
                i2 += 1
         
        child1.calc_id()
        child2.calc_id()

        result = []
        if child1.ID not in self.populationID:
            result.append(child1)

        if child2.ID not in self.populationID:
            result.append(child2)

        return result

    def mutation(self, p, sim, to_complete):
        '''
        :param p:
        :param sim:
        :param to_complete:
        :return:
        '''
        hard_constrained = [p.genes[i] for i in p.base]
        switches = list(itertools.combinations(range(len(p.genes)), 2))

        while len(switches) > 0:
            move = random.choice(switches)
            order = deepcopy(p.genes)
            temp = order[move[0]]
            order[move[0]] = order[move[1]]
            order[move[1]] = temp
            sim.setup()
            sim.init_disablements(order, to_complete)
            error = sim.execute_alternative(order, to_complete)
            if not error:
                # solution is feasible
                base = []
                for i in range(len(order)):
                    if order[i] in hard_constrained:
                        base.append(i)
                return [Chromosome(order, base, self.initialDurationEV)]
        return []


    def mutation_order_based(self, p):
        """
        Order based mutation. It leaves base positions as they are in a parent and randomly permutes
        non-base methods.

        Args:
            parent - Chromosome instance
            tree - taemsTree instance that holds the task structure
        """

        nonBase = list(set(p.genes) - set(p.get_base()))

        child = Chromosome(["" for i in range(len(p.genes))], p.base, self.initialDurationEV)

        for i in range(len(child.base)):
            child.genes[child.base[i]] = p.genes[p.base[i]]

        nonBase = GeneticAlgorithm.permutation(nonBase)
        i1 = 0
        for i in range(len(p.genes)):
            if len(child.genes[i]) == 0:
                child.genes[i] = nonBase[i1]
                i1 += 1
        
        child.calc_id()

        if child.ID not in self.populationID:
            return [child]

        return []

    def mutation_position_based(self, p):
        """
        Position based mutation. Order of methods stays as it is in a parent, new positions of
        base elements are randomly chosen.

        Args:
            parent - Chromosome instance
            tree - taemsTree instance that holds the task structure
        """

        base = random.sample(range(len(p.genes)), len(p.base))
        base.sort()
        child = Chromosome(["" for _ in range(len(p.genes))], base, self.initialDurationEV)

        for i in range(len(child.base)):
            child.genes[child.base[i]] = p.genes[p.base[i]]

        i1 = 0
        for i in range(len(p.genes)):
            if len(child.genes[i]) == 0:
                while i1 in p.base:
                    i1 += 1
                child.genes[i] = p.genes[i1]
                i1 += 1
        
        child.calc_id()

        if child.ID not in self.populationID:
            return [child]

        return []


class Chromosome(object):
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
    def __init__(self, genes, base, method_duration):
        """
        Args:
             genes - an ordered list of methods
             base - a list of positions of base methods
             initialDurationEV - initial expected value for every methods duration (before they are scheduled)
        """
        self.genes = deepcopy(genes)
        self.base = deepcopy(base)
        # self.fitness = 0
        self.total_tardiness = 0
        self.durationEV = deepcopy(method_duration)
        self.schedule = []
        self.ID = id
        self.scheduleDuration = 0
        self.tardy_jobs = {}

    def __str__(self):
        output = ""
        for i in range(len(self.genes)):
            if i in self.base:
                output += '\033[92m' + str(self.genes[i]) + '\033[0m '
            else:
                output += str(self.genes[i]) + ' '
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
        for method in self.genes:

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
        for x in self.genes:
            string += x

        self.ID = md5(string).digest()

    def get_id(self):
        return self.ID

    def get_base(self):
        """
        A method that returns chromosome base methods - elements on self.base positions.
        """
        baseMethods = []
        for x in self.base:
            baseMethods.append(self.genes[x])

        return baseMethods