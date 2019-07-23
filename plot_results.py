
import os
import matplotlib.pyplot as plt


def load_file(directory, filename):
    f = open(os.path.abspath(os.path.join(directory, filename)), 'r')
    results = []
    iter_found = []

    for item in f.readlines():
        line = item.strip().split(" ")
        if len(line) == 0:
            continue
        results.append(float(line[0]))
        iter_found.append(int(line[1]))
    return [results, iter_found]


if __name__ == "__main__":
    [ga_results, ga_iter] = load_file("./Results/", "ga_results.txt")
    [tabu_results, tabu_iter] = load_file("./Results/", "tabu_results_2.txt")
    [sa_results,sa_iter] = load_file("./Results/", "sa_results_3.txt")
    [aco_results, aco_iter] = load_file("./Results/", "aco_results.txt")

    data = [ga_results, tabu_results, sa_results, aco_results]
    data2 = [ga_iter, tabu_iter, sa_iter, aco_iter]
    plt.boxplot(data2)
    plt.title('Performance of scheduling algorithms')
    plt.xticks([1, 2, 3, 4], ['GA', 'tabu search', 'SA', 'ACO'])

    plt.show()


