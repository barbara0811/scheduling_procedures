
import os


def load_file(directory, filename):
    f = open(os.path.abspath(os.path.join(directory, filename)), 'r')
    tasks = []
    pt = {}  # duration
    rt = {}  # release date
    dt = {}  # due date
    state = 0
    for item in f.readlines():
        line = item.strip()
        if len(line) == 0:
            continue
        if line == "TASKS:":
            state = 1
            continue
        # tasks
        if state == 1:
            tasks.append(line.split(",")[0])
            x = line.split(",")
            pt[tasks[-1]] = int(x[1])
            if len(x) > 2:
                rt[tasks[-1]] = int(x[2])
            else:
                rt[tasks[-1]] = -1
            if len(x) > 3:
                dt[tasks[-1]] = int(x[3])
            else:
                dt[tasks[-1]] = -1

    return [tasks, pt, rt, dt]


if __name__ == "__main__":
    [t, pt, rt, dt] = load_file("./Input/", "alternative_1.txt")

    print("Tasks: " + str(t))
    print("Task processing time: " + str(pt))
    print("Release date: " + str(rt))
    print("Due date: " + str(dt))

