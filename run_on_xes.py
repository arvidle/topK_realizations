import topK_realizations.log_generation as log_generation
import time
import itertools
from topK_realizations.algorithm import realization_ranking
from topK_realizations.eval_measures import get_total_number_of_realizations
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
from functools import reduce

#n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_hr_data_anonymize_recognition_results.xes")
#n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_hr_data_anonymize_recognition_results.xes", "uncertainty:classification_prob_windows_start-15_end-15_action_1")
#n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_hr_data_anonymize_recognition_results.xes", "uncertainty:classification_prob_windows_start-7_end-41_action_1")
#n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_ptp_data_anonymize_recognition_results.xes", "uncertainty:classification_prob_windows_start-7_end-41_action_1")
n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_ptp_data_anonymize_recognition_results.xes", "uncertainty:classification_prob_windows_start-7_end-41_action_1")
#n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_ptp_data_anonymize_recognition_results.xes", "uncertainty:name")

#ul = log_generation.simulate_from_log(ul, 10, 0.5)

timing: list[tuple[int, float]] = []

K = 10000
#uncertain_log2 = gen_uncertain_log_simple(100)

print(f"Total number of realizations for this uncertain log: {get_total_number_of_realizations(ul):e}")

start = time.time()
k = 0

logs = []
raw_probabilities = []
distances = []

#print(f"Calculating the top-{K} realizations")
for current_probability, current_log_realization, current_realization in tqdm(itertools.islice(realization_ranking(ul, yield_realization=True), K), total=K, desc=f"Calculating the top-{K} realizations"):
    raw_probabilities.append(current_probability)
    distances.append(reduce(lambda x, y: x + 1 if y else x, list(zip(*current_realization))[0]))
    step = time.time()
    timing.append((k, step-start))
    k += 1

# Get expected value for raw probability if the probabilities were equally distributed
expected_probability_uniform = np.full_like(raw_probabilities, (1 / n_activities)**(len(ul)))


print(f"Sum of raw probabilities: {sum(raw_probabilities):.5e}")

max_probability = raw_probabilities[0]


def count_simple_event_logs(logs, sort_key):
    # Convert dicts to lists of tuples
    logs_as_list_of_tuples = [list(l.items()) for l in logs]
    # Sort each list of tuples by the 0-th element (keys)
    logs_as_sorted_list_of_tuples = [sorted(l, key=sort_key) for l in logs_as_list_of_tuples]
    # This gives us a list of list of tuples
    # To make this hashable, convert to tuple of tuples of tuples
    logs_as_tuple_of_tuples = tuple([tuple(l) for l in logs_as_sorted_list_of_tuples])
    # Then, apply a Counter
    return Counter(logs_as_tuple_of_tuples)


def sum_simple_event_log_probabilities(logs, probabilities, sort_key):
    res = defaultdict(lambda: 0)
    # Convert dicts to lists of tuples
    logs_as_list_of_tuples = [list(l.items()) for l in logs]
    # Sort each list of tuples by the 0-th element (keys)
    logs_as_sorted_list_of_tuples = [sorted(l, key=sort_key) for l in logs_as_list_of_tuples]
    # This gives us a list of list of tuples
    # To make this hashable, convert to tuple of tuples of tuples
    logs_as_tuple_of_tuples = tuple([tuple(l) for l in logs_as_sorted_list_of_tuples])
    for probability, simple_log in zip(probabilities, logs_as_tuple_of_tuples):
        res[simple_log] += probability

    return res

timing_dict = dict(timing)
runtime_plot_data = []
p1_plot_data = []
cp_plot_data = []
mean_distance_plot_data = []
runtime_plot_data = []
#plot_steps = np.concatenate((np.arange(0, 100, 1), np.arange(100, 1000, 10), np.arange(1000, 1000000, 1000)))
plot_steps = np.concatenate((np.arange(0, 100, 1), np.arange(100, 10000, 10)))

print(f"Total runtime: {timing[-1][1]}")

for i in plot_steps:
    p1 = raw_probabilities[0]
    p1_plot_data.append(p1)
    cp = sum(raw_probabilities[:i+1])
    cp_plot_data.append(cp)
    mean_distance = sum(distances[:i+1]) / len(distances[:i+1])
    mean_distance_plot_data.append(mean_distance)
    runtime = timing[i][1]
    runtime_plot_data.append(runtime)

np.savez("./evaluation/results/hr.npz", steps=plot_steps, p1_plot_data=p1_plot_data, cp_plot_data=cp_plot_data, mean_distance_plot_data=mean_distance_plot_data, runtime_plot_data=runtime_plot_data)
