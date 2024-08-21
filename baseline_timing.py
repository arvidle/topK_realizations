from topK_realizations.algorithm import realization_ranking_baseline
import topK_realizations.log_generation as log_generation
import time
import itertools

K = 10000
# Test on simulated logs
loglens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]#[1, 2, 5, 10, 20, 50]

for loglen in loglens:
    simulated_log = log_generation.simulate_randomly(loglen, 0.3, 0.3, 3)
    start = time.time()
    for res in itertools.islice(realization_ranking_baseline(simulated_log), K):
        pass
    end = time.time()
    print(f"Calculating top-{K} realizations of a log with length {loglen} took {end - start} seconds")

# Test on XES log
#n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_ptp_data_anonymize_recognition_results.xes", "uncertainty:classification_prob_windows_start-7_end-41_action_1")

#print(n_activities)

#start = time.time()
#for res in itertools.islice(realization_ranking_baseline(ul), K):
#    pass
#end = time.time()

#print(f"Calculating top-{K} realizations took {end - start} seconds")