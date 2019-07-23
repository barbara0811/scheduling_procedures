
__author__ = "barbanas"

import copy
import taems
import helper_functions

class Simulator(object):
    '''
    Simulates the execution of a schedule. 
    Applies the effects of method execution - interrelationships, resources...
    Calculates the quality, duration and cost probability distributions of the schedule.

    Attributes:
        taemsTree
        numberOfCompletedSubtasks - dictionary {task label : number of its completed subtasks}
        completedTasks - a set of completed tasks
        disabledMethods - a set of currently disabled methods
        taskHeuristicRating - dictionary {method label : rating (float number)}
    '''
    def __init__(self, tree):
        
        self.taemsTree = copy.deepcopy(tree)

        self.numberOfCompletedSubtasks = {}
        for task in self.taemsTree.tasks.values():
            if type(task) is taems.TaskGroup:
                self.numberOfCompletedSubtasks[task.label] = 0
                for subtask in task.subtasks:
                    if subtask not in self.taemsTree.tasks.keys():
                        continue
                    #if self.taemsTree.tasks[subtask].subtasks is None:
                        #if self.taemsTree.tasks[subtask].nonLocal:
                        #    continue
                        #else:
                            #self.numberOfCompletedSubtasks[task.label] += 1
                    #else:
                    self.numberOfCompletedSubtasks[task.label] += 1

        #print self.numberOfCompletedSubtasks
        #raw_input("\n###\n")
        self.completedTasks = []
        self.taskOutcome = {}
        self.taskEndTime = {}

    def clone(self):
        s = Simulator(self.taemsTree)
        s.numberOfCompletedSubtasks = copy.deepcopy(self.numberOfCompletedSubtasks)
        s.completedTasks = copy.deepcopy(self.completedTasks)
        s.taskOutcome = copy.deepcopy(self.taskOutcome)
        s.taskEndTime = copy.deepcopy(self.taskEndTime)
        return s
        
    def execute_schedule(self, schedule, nonLocal, time):
        newSchedule = []
        methodList = [i[0] for i in schedule]
        
        for item in nonLocal:
            self.execute_non_local_task(item)
            
        for item in schedule:
            if item[0] == "slack":
                if len(newSchedule) == 0:
                    duration = item[2]
                else:
                    duration = item[2] - newSchedule[-1][2]
            else:
                duration = self.taemsTree.tasks[item[0]].DurationEV
                self.execute_task(item[0], methodList, time + duration, True)
            newSchedule.append([item[0], time, time + duration])
            time += duration

        self.evaluate_schedule()
        return newSchedule
    
    def calc_task_end_times(self, schedule):
        
        for item in schedule:
            if item[0] == "slack":
                continue
            else:
                self.execute_task(item[0], [], item[2], False)

    def execute_non_local_task(self, task):
        #print ".."
        #print task

        # if the task is Method
        if self.taemsTree.tasks[task].subtasks is None:
            if self.taemsTree.tasks[task].nonLocal:
                self.completedTasks.append(task)

        # if root task -> return
        if len(self.taemsTree.tasks[task].supertasks) == 0:
            return

        parentLabel = self.taemsTree.tasks[task].supertasks[0]
        self.numberOfCompletedSubtasks[parentLabel] -= 1
        #print self.numberOfCompletedSubtasks[parentLabel]
        if parentLabel in self.completedTasks:
            return

        qaf = self.taemsTree.tasks[parentLabel].qaf
        # if execution of only one subtask grants the execution of the subtask
        if qaf == 'q_max' or qaf == 'q_sum' or self.numberOfCompletedSubtasks[parentLabel] == 0:
            self.execute_non_local_task(parentLabel)
        #raw_input(",,")

    def execute_task(self, task, alternative, endTime, activateIR):
        '''
        Description.

        Args:
            mathodLabel - string
            time - float, task start time
        '''
        #print ".."
        #print task
        if task in self.completedTasks:
            return
        
        # activate interrelationships
        if activateIR:
            self.activate_IR(task, alternative, endTime)
        
        # update the list of completed tasks
        self.completedTasks.append(task)
        
        # method
        if self.taemsTree.tasks[task].subtasks is None:
            self.taskEndTime[task] = endTime

        # if root task -> return
        if len(self.taemsTree.tasks[task].supertasks) == 0:
            return

        parentLabel = self.taemsTree.tasks[task].supertasks[0]
        #print "parent   " + parentLabel
        self.numberOfCompletedSubtasks[parentLabel] -= 1
        #print self.numberOfCompletedSubtasks[parentLabel]
        if parentLabel in self.completedTasks:
            return

        qaf = self.taemsTree.tasks[parentLabel].qaf
        # if execution of only one subtask grants the execution of the subtask
        if  qaf == 'q_max' or qaf == 'q_sum' or self.numberOfCompletedSubtasks[parentLabel] == 0:
            if parentLabel in self.taskEndTime.keys():
                if qaf == 'q_max':
                    self.taskEndTime[parentLabel] = min(self.taskEndTime[parentLabel], self.taskEndTime[task]) #TODO adjust this part! -- if multiple children are executed -- end time = max(..)
                else:
                    self.taskEndTime[parentLabel] = max(self.taskEndTime[parentLabel], self.taskEndTime[task])
            else:
                self.taskEndTime[parentLabel] = self.taskEndTime[task]

            self.execute_task(parentLabel, alternative, endTime, activateIR)

    def activate_IR(self, task, alternative, taskEndTime):
        '''
        Activates the effects of interrelationship and modifies soft heuristic ratings.

        Args:
            task - string, a task label
            time - float number, start of ir.From task's execution
        '''
        insufficientResource = []
        for ir in self.taemsTree.IRs.values():
            if ir.From != task:
                continue
            
            # hard and soft IR (0 - enables, 1 - disables, 2 - facilitates, 3 - hinders)
            if ir.type < 4:
                # if ir.To is not my task -> continue
                if ir.To in alternative and ir.To not in self.completedTasks:
                    ir.activate(self.taemsTree, taskEndTime)

            # consumes
            if ir.type == 5:
                if ir.To in self.taemsTree.resources.keys():
                    ir.activate(self.taemsTree)
                    self.taemsTree.resources[ir.To].checkSufficiency()
                    if self.taemsTree.resources[ir.To].isSufficient == False:
                        insufficientResource.append(ir.To)
                        
        if len(insufficientResource) > 0:
            for resource in insufficientResource:
                irList = []
                for IR in self.taemsTree.IRs.values():
                    if resource == IR.From and ir.To in self.taemsTree.tasks.keys():
                        irList.append(IR)
                
                if len(irList) == 0:
                    continue
                
                for ir in irList:
                    ir.activate(self.taemsTree, 0)

    def evaluate_schedule(self):
        '''
        Evaluates a schedule. Calculates quality, duration and cost probability distributions and
        quality, duration and cost expected values.

        Returns:
            A list [quality distribution, duration distr., cost distr., quality EV, duration EV, cost EV]
        '''
        [Q, D, C] = self.calc_QDC(self.taemsTree.rootTask[0])

    def calc_QDC(self, task):
        '''
        Recursive function. Calculates task's quality, duration and cost probability distributions using
        subtask's outcomes and quality accumulation function.

        Args:
            task - string, task's label

        Returns:
            list - [quality distribution, duration distr., cost distr.]
        '''

        if task not in self.completedTasks:
            return None

        if type(self.taemsTree.tasks[task]) is taems.Method:
            # self.taemsTree.tasks[task].calcExpectations()
            q = self.taemsTree.tasks[task].outcome[0]
            d = self.taemsTree.tasks[task].outcome[1]
            c = self.taemsTree.tasks[task].outcome[2]
            self.taskOutcome[task] = [q, d, c]
            return [q, d, c]

        qaf = self.taemsTree.tasks[task].qaf

        qualityDistributions = []
        durationDistributions = []
        costDistributions = []

        for child in self.taemsTree.tasks[task].subtasks:
            if child in self.taskOutcome.keys():
                childQDC = self.taskOutcome[child]
            else:
                childQDC = self.calc_QDC(child)

            if childQDC == None:
                continue

            qualityDistributions.append(childQDC[0])
            durationDistributions.append(childQDC[1])
            costDistributions.append(childQDC[2])
            
        fun = ""
        if qaf == "q_min":
            fun = "min"
        elif qaf == "q_max":
            fun = "max"
        elif qaf == 'q_sum_all' or qaf == 'q_sum' or qaf == 'q_seq_sum_all':
            fun = "sum"

        Q = helper_functions.cartesianProductOfDistributions(qualityDistributions, fun)
        C = helper_functions.cartesianProductOfDistributions(costDistributions, "sum")
        D = helper_functions.cartesianProductOfDistributions(durationDistributions, "sum")

        self.taskOutcome[task] = [Q, D, C]
        return [Q, D, C]

class LightSimulator(object):
    '''
    Simulates the execution of a schedule. 
    Applies the effects of method execution - HARD PRECEDENCE CONSTRAINTS ONLY.
    
    Attributes:
        initialTree - taemsTree instance -> no effects are applied to it
        taemsTree - taemsTree instance used for simulation
        numberOfCompletedSubtasks - dictionary {task label : number of its completed subtasks}
        completedTasks - a set of completed tasks
        disabledMethods - a set of currently disabled methods
        hardConstrainedTasks - a set of tasks that are a source or a target of hard task interrelationship
    '''

    def __init__(self, tree):
        self.taemsTree = tree
        self.setup()

    def setup(self):
        """
        """
        self.numberOfCompletedSubtasks = {}
        for task in self.taemsTree.tasks.values():
            if type(task) is taems.TaskGroup:
                self.numberOfCompletedSubtasks[task.label] = 0
                for subtask in task.subtasks:
                    if subtask not in self.taemsTree.tasks.keys():
                        continue
                    '''
                    if self.taemsTree.tasks[subtask].subtasks is None:
                        if self.taemsTree.tasks[subtask].nonLocal:
                            continue
                        else:
                            self.numberOfCompletedSubtasks[task.label] += 1
                    else:
                    '''
                    self.numberOfCompletedSubtasks[task.label] += 1

        self.disablements = {}
        self.completedTasks = []
        self.disabledMethods = set()
        self.hardConstrainedTasks = set()

    def init_disablements(self, alternative, tasksToComplete):
        """
        A method that disables methods in IREnables.To and IRDisables.From to 
        enforce hard precedence constraints.

        Args:
            alternative - unordered list of methods
        """
        for ir in self.taemsTree.IRs.values():
            if ir.To in tasksToComplete and ir.From in tasksToComplete:
                # disable all enables IR
                if ir.type == 0:
                    '''
                    if self.taemsTree.tasks[ir.From].subtasks is None:
                        if self.taemsTree.tasks[ir.From].nonLocal:
                            continue
                    '''
                    self.disable_task(ir.To)
                    self.add_hard_constrained_methods(ir.To, alternative)
                    self.add_hard_constrained_methods(ir.From, alternative)

                # disable all disables IR
                if ir.type == 1:
                    '''
                    if self.taemsTree.tasks[ir.To].subtasks is None:
                        if self.taemsTree.tasks[ir.To].nonLocal:
                            continue
                    '''
                    self.disable_task(ir.From)
                    self.add_hard_constrained_methods(ir.To, alternative)
                    self.add_hard_constrained_methods(ir.From, alternative)
            # limits
            if ir.type == 6:
                if ir.From in self.taemsTree.resources.keys() and ir.To in tasksToComplete:
                    self.taemsTree.resources[ir.From].checkSufficiency()
                    if self.taemsTree.resources[ir.From].isSufficient == False:
                        ir.activate(self.taemsTree, 0)
                        if 1.0 in ir.quality_power:
                            if ir.quality_power[1.0] == 1.0:
                                self.disable_task(ir.To)

    def add_hard_constrained_methods(self, task, alternative):

        if task not in self.taemsTree.tasks.keys():
            return

        # a method
        if self.taemsTree.tasks[task].subtasks is None:
            '''
            if task in alternative:
                self.hardConstrainedTasks.add(task)
            '''
            self.hardConstrainedTasks.add(task)
            return

        for subtask in self.taemsTree.tasks[task].subtasks:
            self.add_hard_constrained_methods(subtask, alternative)

    def disable_task(self, task):
        """
        A method that disables a task and all of its subtasks.

        Args:
            task - task label
        """
        if task not in self.taemsTree.tasks.keys():
            return
        
        # method
        if self.taemsTree.tasks[task].subtasks is None:
            #if self.taemsTree.tasks[task].nonLocal:
            #    return
        
            if task in self.disablements.keys():
                self.disablements[task] += 1
            else:
                self.disablements[task] = 1
            self.disabledMethods.add(task)

        else:
            for subtask in self.taemsTree.tasks[task].subtasks:
                self.disable_task(subtask)

    def enable_task(self, task):
        """
        A method that enables a task and all of its subtasks.

        Args:
            task - task label
        """
        if task not in self.taemsTree.tasks.keys():
            return
        
        # method
        if self.taemsTree.tasks[task].subtasks is None:
            #if self.taemsTree.tasks[task].nonLocal:
            #    return
            if task in self.disabledMethods:
                self.disablements[task] -= 1
                if self.disablements[task] == 0:
                    self.disabledMethods.remove(task)

        else:
            for subtask in self.taemsTree.tasks[task].subtasks:
                #if subtask in 
                self.enable_task(subtask)

    def get_enabled_tasks(self, alternative):
        """
        A method that returns a set of enabled tasks in list of methods.

        Args:
            alternative - a list of methods
        """
        return set(alternative) - self.disabledMethods

    def execute_alternative(self, alternative, tasksToComplete):
        for task in alternative:
            error = self.execute_task(task, tasksToComplete)
            if error:
                return 1
        return 0

    def execute_task(self, task, tasksToComplete):
        """
        A method that simulates the effects of method execution (only hard precedence constraints and completed 
        task list).

        Args:
            task - a task to execute
        """
        if task in self.completedTasks:
            return 0

        if task in self.disabledMethods:
            return 1

        # activate hard IRs
        self.activate_IR(task, tasksToComplete)

        # update the list of completed tasks
        self.completedTasks.append(task)
        
        # if task is non local
        #if task not in self.taemsTree.tasks.keys():
        #    return

        if len(self.taemsTree.tasks[task].supertasks) == 0:
            return 0

        parentLabel = self.taemsTree.tasks[task].supertasks[0]

        self.numberOfCompletedSubtasks[parentLabel] -= 1

        if parentLabel in self.completedTasks:
            return 0

        qaf = self.taemsTree.tasks[parentLabel].qaf
        # if execution of only one subtask grants the execution of the subtask
        if qaf == 'q_max' or qaf == 'q_sum' or self.numberOfCompletedSubtasks[parentLabel] == 0:
            self.execute_task(parentLabel, tasksToComplete)
            '''
            if len(self.taemsTree.tasks[parentLabel].supertasks) > 0:
                parentLabel = self.taemsTree.tasks[parentLabel].supertasks[0]
                taskCompleted = True
                '''

    def activate_IR(self, task, tasksToComplete):
        """
        A task that creates the effects of hard precedence constraints - enables and disables methods.

        Args:
            task - a label of task that is being executed
        """
        insufficientResource = []
        for ir in self.taemsTree.IRs.values():
            if ir.From == task:
                if ir.type < 2 and ir.To in tasksToComplete and ir.To not in self.completedTasks:
                    # hard IR (0 - enables, 1 - disables)
                    if ir.type == 0:
                        self.enable_task(ir.To)
    
                    if ir.type == 1:
                        self.disable_task(ir.To)

                # consumes
                if ir.type == 5:
                    if ir.To in self.taemsTree.resources.keys():
                        self.taemsTree.resources[ir.To].state
                        ir.activate(self.taemsTree)
                        self.taemsTree.resources[ir.To].checkSufficiency()
                        self.taemsTree.resources[ir.To].state
                        if self.taemsTree.resources[ir.To].isSufficient == False:
                            insufficientResource.append(ir.To)
                
            if ir.To == task:
                # disables -> task in ir.From in IRDisables was initially disabled, so it has to be enabled
                if ir.type == 1:
                    self.enable_task(ir.From)
                    
        if len(insufficientResource) > 0:
            for resource in insufficientResource:
                irList = []
                for IR in self.taemsTree.IRs.values():
                    if resource == IR.From and ir.To in tasksToComplete:
                        irList.append(IR)
                
                if len(irList) == 0:
                    continue
                
                for ir in irList:
                    ir.activate(self.taemsTree, 0)
                    if 1.0 in ir.quality_power:
                        if ir.quality_power[1.0] == 1.0:
                            self.disable_task(ir.To)
            
