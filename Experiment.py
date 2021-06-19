import numpy as np
import strlearn as sl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.naive_bayes import GaussianNB
from strlearn.ensembles import OOB, UOB, SEA, OnlineBagging
from scipy.stats import ranksums
from tabulate import tabulate
from OurOOB import OurOOB

clfs = {
    'UOB': UOB(base_estimator=GaussianNB(), n_estimators=10),
    'OurOOB': OurOOB(base_estimator=GaussianNB(), n_estimators=10),
    'OOB': OOB(base_estimator=GaussianNB(), n_estimators=10),
    'OB': OnlineBagging(base_estimator=GaussianNB(), n_estimators=10),
    'SEA': SEA(base_estimator=GaussianNB(), n_estimators=10)
}

clf_names = [
    "UOB",
    "OurOOB",
    "OOB",
    "OB",
    "SEA"
]

metrics_names = ["G-mean", "F1 Score"]

metrics = ["G-mean", "F1 Score"]
metrics_g = [sl.metrics.geometric_mean_score_1]
metrics_f1 = [sl.metrics.f1_score]

n_chunks_value = 100

cm = LinearSegmentedColormap.from_list(
    "lokomotiv", colors=[(0.3, 0.7, 0.3), (0.7, 0.3, 0.3)]
)

chunks_plotted = np.linspace(0, n_chunks_value - 1, 8).astype(int)


def plot_stream(stream, filename="foo", title=""):
    fig, ax = plt.subplots(1, len(chunks_plotted), figsize=(14, 2.5))

    j = 0
    for i in range(n_chunks_value):
        X, y = stream.get_chunk()
        if i in chunks_plotted:
            ax[j].set_title("Chunk %i" % i)
            ax[j].scatter(X[:, 0], X[:, 1], c=y, cmap=cm, s=10, alpha=0.5)
            ax[j].set_ylim(-4, 4)
            ax[j].set_xlim(-4, 4)
            ax[j].set(aspect="equal")
            j += 1

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig("%s.png" % filename, transparent=True)


def generating_streams(random_state_value, number):

    stream = sl.streams.StreamGenerator(n_chunks=n_chunks_value,
                                        chunk_size=500,
                                        n_classes=2,
                                        n_features=10,
                                        weights=[0.3, 0.7],
                                        random_state=random_state_value)

    evaluator_g = sl.evaluators.TestThenTrain(metrics_g)
    evaluator_g.process(stream, clfs.values())

    stream.reset()

    evaluator_f1 = sl.evaluators.TestThenTrain(metrics_f1)
    evaluator_f1.process(stream, clfs.values())

    stream.reset()
 
    plot_stream(stream, f"static-imbalanced{number}", "Niezbalansowany statyczny strumień danych dla IR = 30%")

    return evaluator_g.scores, evaluator_f1.scores


random_state_list = [10, 24, 16, 124, 421, 661, 902, 1235, 2138, 3331]

full_scores_g = []
full_scores_f1 = []

i = 0
for r, random_state in enumerate(random_state_list):
    scores_g, scores_f1 = generating_streams(random_state, r)
    if i == 0:
        full_scores_g = scores_g
        full_scores_f1 = scores_f1
        i = 1
    else:
        arr = np.append(full_scores_g, scores_g, axis=0)
        full_scores_g = arr
        arr = np.append(full_scores_f1, scores_f1, axis=0)
        full_scores_f1 = arr


# saving results (evaluator scores) to file
def save_to_file(_full_scores, _metric):
    np.save(f'results_{_metric}', _full_scores)


save_to_file(full_scores_g, 'g')
save_to_file(full_scores_f1, 'f1')

### Data analisis ###

# reading_result_from_file
scores_g = np.load('results_g.npy', allow_pickle=True)
scores_f1 = np.load('results_f1.npy', allow_pickle=True)
# print("\n\nScores:\n", scores)

fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
for m, metric in enumerate(metrics):
    ax[m].set_title(metrics_names[m])
    ax[m].set_ylim(0, 1)
    if m == 0:
        for i, clf in enumerate(clfs):
            ax[m].plot(scores_g[i, :, 0], label=clf_names[i])
    else:
        for i, clf in enumerate(clfs):
            ax[m].plot(scores_f1[i, :, 0], label=clf_names[i])
    plt.title("Przetwarzanie strumieniowe dla IR = 30%")
    plt.ylabel("Metric")
    plt.xlabel("Chunk")
    ax[m].legend()
plt.savefig("comparison.png", transparent=True)


# mean scores
def mean_calculate(mean):
    means_list = []
    for j in range(0, len(clfs)):
        clf_mean = []
        for i in range(j, len(mean) - 1, len(clfs)):
            arr = np.append(clf_mean, mean[i], axis=0)
            clf_mean = arr
        means = np.mean(clf_mean)
        means_list.append(means)
    return means_list


# showing mean
mean_array_g = np.mean(scores_g, axis=1)
mean_g = mean_calculate(mean_array_g)
print("\n\nMean for G-mean metric:\n", mean_g)

mean_array_f1 = np.mean(scores_f1, axis=1)
mean_f1 = mean_calculate(mean_array_f1)
print("\n\nMean for F1 Score metric:\n", mean_f1)

# std scores
def std_calculate(std):
    std_list = []
    for j in range(0, len(clfs)):
        std_clf = []
        for i in range(j, len(std) - 1, len(clfs)):
            arr = np.append(std_clf, std[i], axis=0)
            std_clf = arr
        stds = np.std(std_clf)
        std_list.append(stds)
    return std_list


# showing std
std_array_g = np.std(scores_g, axis=1)
std_g = std_calculate(std_array_g)

std_array_f1 = np.std(scores_f1, axis=1)
std_f1 = std_calculate(std_array_f1)
# print("\n\nStd:\n", std_sudden)


### Results ###

def show_results(mean, std):
    for clf_id, clf_name in enumerate(clfs):
        print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))


print("\n\nResults for G-mean metric:")
show_results(mean_g, std_g)

print("\n\nResults for F1 Score metric:")
show_results(mean_f1, std_f1)

### Statistical analisis ###


# creating results tabels
def create_result_tables(_scores):
    # defining alfa value
    alfa = .05

    # setting up w_statistic and p_value array filled up with zeros
    w_statistic = np.zeros((len(clfs), len(clfs)))
    p_value = np.zeros((len(clfs), len(clfs)))

    # creating t_statistic and p_value matrices
    for i in range(len(clfs)):
        for j in range(len(clfs)):
            w_statistic[i, j], p_value[i, j] = ranksums(_scores[i], _scores[j])
    print("\nw-statistic:\n", w_statistic, "\n\np-value:\n", p_value)

    headers = ["UOB", "ourOOB", "OOB", "OB", "SEA"]
    names_column = np.array([["UOB"], ["ourOOB"], ["OOB"], ["OB"], ["SEA"]])
    w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
    w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f", tablefmt="github")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f", tablefmt="github")
    print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

    # advantage matrics
    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[w_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("\n\nAdvantage:\n\n", advantage_table)

    # significance table
    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("\n\nStatistical significance (alpha = 0.05):\n\n", significance_table)

    # final results
    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("\n\nStatistically significantly better:\n\n", stat_better_table)


# final results
print("\n\n==============================================================================================")
print("Results for G-mean:")
create_result_tables(scores_g)
print("\n\n==============================================================================================")
print("Results for F1 Score:")
create_result_tables(scores_f1)
