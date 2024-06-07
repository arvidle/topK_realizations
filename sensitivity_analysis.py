from topK_realizations.algorithm import realization_ranking
from tqdm.auto import tqdm
import numpy as np
import topK_realizations.log_generation as log_generation
import time
import itertools
from topK_realizations.eval_measures import check_split_point
from datetime import datetime
import os
import pickle
import math
from plot_sensitivity_analysis import statistics_table, plot_results
from functools import reduce

K = 10000
REPEAT = 10
BASE_OUT_DIR = "evaluation/results/sensitivity_analysis/"
if not os.path.exists(BASE_OUT_DIR):
    os.makedirs(BASE_OUT_DIR)


def get_statistics(uncertain_log, K, timing_step=100, verbose=False):
    timing: list[tuple[int, float]] = []
    start = time.time()
    k = 0
    raw_probabilities = []
    distances = []

    for current_probability, current_log_realization, current_realization in (
            itertools.islice(realization_ranking(uncertain_log, yield_realization=True), K)):
        k += 1

        raw_probabilities.append(current_probability)
        distances.append(reduce(lambda x, y: x + 1 if y else x, list(zip(*current_realization))[0]))

        if k % timing_step == 0:
            step = time.time()
            timing.append((k, step - start))

    timing.append((k, time.time() - start))


    point01 = check_split_point(raw_probabilities, 0.1)
    point05 = check_split_point(raw_probabilities, 0.5)
    cum_probability = sum(raw_probabilities)

    if verbose:
        print(f"Ten percent point for top-{K} is at {point01}")
        print(f"Halfway point for top-{K} is at {point05}")
        print(f"Cumulative probability for top-{K} is {cum_probability}")

    return {
        "point01": point01,
        "point05": point05,
        "cum_probability": cum_probability,
        "timing": timing,
        "probabilities": raw_probabilities,
        "distances": distances,
        "mean_distance": sum(distances) / len(distances),
        "median_distance": distances[math.floor(len(distances) / 2)]
    }


def check_runtime(uncertain_log, K, timing_step=100, verbose=False):
    timing: list[tuple[int, float]] = []
    start = time.time()
    k = 0

    for _ in itertools.islice(realization_ranking(uncertain_log, yield_realization=True), K):
        k += 1
        if k % timing_step == 0:
            step = time.time()
            timing.append((k, step - start))

    timing.append((k, time.time() - start))

    return timing

#for thres in np.arange(0.0, 1.501, 0.02):
#    simulated_log = log_generation.simulate_from_log(ul, 10, thres)
#    get_statistics(simulated_log, K)


def check_runtime_over_n_cases(ul, n_cases_min, n_cases_max, n_cases_step, K, verbose=False):
    """
    TODO: It would be cleaner to check the runtime over a set number of events instead of cases...
    :param ul:
    :param n_cases_min:
    :param n_cases_max:
    :param n_cases_step:
    :param K:
    :return:
    """

    timing = []
    for n_cases in range(n_cases_min, n_cases_max, n_cases_step):
        uncertain_log = log_generation.simulate_from_log(ul, n_cases, 0.5)
        start = time.time()
        k = 0
        for _, _ in itertools.islice(realization_ranking(uncertain_log), K):
            k += 1
        end = time.time()
        timing.append((len(uncertain_log), end - start))
        if verbose:
            print((len(uncertain_log), end - start))

    return timing


def sensitivity_analysis():
    out_dir = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
    full_out_dir = os.path.join(BASE_OUT_DIR, out_dir)
    os.mkdir(full_out_dir)

    # n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_hr_data_anonymize_recognition_results.xes")
    # n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_ptp_data_anonymize_recognition_results.xes", "uncertainty:classification_prob_windows_start-7_end-41_action_1")
    #n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_ptp_data_anonymize_recognition_results.xes",
                                                        # "uncertainty:classification_prob_windows_start-15_end-15_action_1")

    for i in tqdm(range(REPEAT), desc=f"Repeating analysis {REPEAT} times", leave=False):
        for n_events_it in tqdm(np.arange(0, 10000, 1000), desc="Varying n_events", leave=False):
            n_events = n_events_it
            for thres in tqdm(np.arange(0.0, 1.01, 0.05), desc="Varying uncertainty threshold", leave=False):
                for n_acts in tqdm(np.arange(2, 6), desc="Varying number of activities in uncertain events", leave=False):
                        simulated_log = log_generation.simulate_randomly(n_events, thres, 0.5)
                        res = get_statistics(simulated_log, K)
                        output_filename = os.path.join(BASE_OUT_DIR, out_dir, f"nevents_{n_events}_thres_{thres}_nacts_{n_acts}_run_{i}.pkl")
                        with open(output_filename, "wb") as file:
                            pickle.dump(res, file)


def sensitivity_analysis_n_acts(n_events, thres, beta, K, n_acts_low, n_acts_high, n_repeats=REPEAT):
    out_dir = datetime.now().strftime("%Y-%d-%m_%H-%M-%S") + f"_analyze_n_acts_with_nevents_{n_events}_thres_{thres}_beta_{beta}"
    full_out_dir = os.path.join(BASE_OUT_DIR, out_dir)
    os.mkdir(full_out_dir)

    # n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_hr_data_anonymize_recognition_results.xes")
    # n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_ptp_data_anonymize_recognition_results.xes", "uncertainty:classification_prob_windows_start-7_end-41_action_1")
    #n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_ptp_data_anonymize_recognition_results.xes",
                                                        # "uncertainty:classification_prob_windows_start-15_end-15_action_1")

    res_wrapper = dict()
    res_wrapper["n_events"] = n_events
    res_wrapper["n_repeats"] = n_repeats
    res_wrapper["n_steps"] = n_acts_high + 1 - n_acts_low
    res_wrapper["thres"] = thres
    res_wrapper["beta"] = beta
    res_wrapper["run_results"] = []
    res_wrapper["K"] = K
    for n_acts in tqdm(np.arange(n_acts_low, n_acts_high + 1), desc="Varying number of activities in uncertain events", leave=False):
        res_runs = dict()
        res_runs["n_acts"] = n_acts
        res_runs["runs"] = []
        for i in range(n_repeats):
            simulated_log = log_generation.simulate_randomly(n_events, thres, beta, n_acts)
            res = get_statistics(simulated_log, K)
            timing = check_runtime(simulated_log, K)
            res["timing"] = timing
            res["run"] = i
            res_runs["runs"].append(res)
        res_wrapper["run_results"].append(res_runs)

    output_filename = os.path.join(BASE_OUT_DIR, out_dir, f"results.pkl")
    with open(output_filename, "wb") as file:
        pickle.dump(res_wrapper, file)

    return res_wrapper, full_out_dir


def sensitivity_analysis_n_events(n_acts, thres, beta, K, n_events_values, n_repeats=REPEAT):
    out_dir = datetime.now().strftime("%Y-%d-%m_%H-%M-%S") + f"_analyze_n_events_with_n_acts_{n_acts}_thres_{thres}_beta_{beta}"
    full_out_dir = os.path.join(BASE_OUT_DIR, out_dir)
    os.mkdir(full_out_dir)

    # n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_hr_data_anonymize_recognition_results.xes")
    # n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_ptp_data_anonymize_recognition_results.xes", "uncertainty:classification_prob_windows_start-7_end-41_action_1")
    #n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_ptp_data_anonymize_recognition_results.xes",
                                                        # "uncertainty:classification_prob_windows_start-15_end-15_action_1")

    res_wrapper = dict()
    res_wrapper["n_acts"] = n_acts
    res_wrapper["n_repeats"] = n_repeats
    res_wrapper["n_steps"] = len(n_events_values)
    res_wrapper["n_events_values"] = n_events_values
    res_wrapper["thres"] = thres
    res_wrapper["beta"] = beta
    res_wrapper["run_results"] = []
    res_wrapper["K"] = K
    for n_events in tqdm(n_events_values, desc="Varying length of the event log (number of events)", leave=False):
        res_runs = dict()
        res_runs["n_events"] = n_events
        res_runs["runs"] = []
        for i in range(n_repeats):
            simulated_log = log_generation.simulate_randomly(n_events, thres, beta, n_acts)
            res = get_statistics(simulated_log, K)
            timing = check_runtime(simulated_log, K)
            res["timing"] = timing
            res["run"] = i
            res_runs["runs"].append(res)
        res_wrapper["run_results"].append(res_runs)

    output_filename = os.path.join(BASE_OUT_DIR, out_dir, f"results.pkl")
    with open(output_filename, "wb") as file:
        pickle.dump(res_wrapper, file)

    return res_wrapper, full_out_dir


def sensitivity_analysis_beta(n_events, n_acts, thres, K, beta_values, n_repeats=REPEAT):
    out_dir = datetime.now().strftime("%Y-%d-%m_%H-%M-%S") + f"_analyze_beta_with_n_events_{n_events}_n_acts_{n_acts}_thres_{thres}"
    full_out_dir = os.path.join(BASE_OUT_DIR, out_dir)
    os.mkdir(full_out_dir)

    # n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_hr_data_anonymize_recognition_results.xes")
    # n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_ptp_data_anonymize_recognition_results.xes", "uncertainty:classification_prob_windows_start-7_end-41_action_1")
    #n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_ptp_data_anonymize_recognition_results.xes",
                                                        # "uncertainty:classification_prob_windows_start-15_end-15_action_1")

    res_wrapper = dict()
    res_wrapper["n_acts"] = n_acts
    res_wrapper["n_repeats"] = n_repeats
    res_wrapper["n_steps"] = len(beta_values)
    res_wrapper["beta_values"] = beta_values
    res_wrapper["thres"] = thres
    res_wrapper["n_events"] = n_events
    res_wrapper["run_results"] = []
    res_wrapper["K"] = K
    for beta in tqdm(beta_values, desc="Varying event probability distribution skewness (beta)", leave=False):
        res_runs = dict()
        res_runs["beta"] = beta
        res_runs["runs"] = []
        for i in range(n_repeats):
            simulated_log = log_generation.simulate_randomly(n_events, thres, beta, n_acts, r_low=0.9, r_high=1.1)
            res = get_statistics(simulated_log, K)
            res["timing"] = check_runtime(simulated_log, K)
            res["run"] = i
            res_runs["runs"].append(res)
        res_wrapper["run_results"].append(res_runs)

    output_filename = os.path.join(BASE_OUT_DIR, out_dir, f"results.pkl")
    with open(output_filename, "wb") as file:
        pickle.dump(res_wrapper, file)

    return res_wrapper, full_out_dir


def sensitivity_analysis_thres(n_events, n_acts, beta, K, thres_values, n_repeats=REPEAT):
    out_dir = datetime.now().strftime("%Y-%d-%m_%H-%M-%S") + f"_analyze_thres_with_n_events_{n_events}_n_acts_{n_acts}_beta_{beta}"
    full_out_dir = os.path.join(BASE_OUT_DIR, out_dir)
    os.mkdir(full_out_dir)

    # n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_hr_data_anonymize_recognition_results.xes")
    # n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_ptp_data_anonymize_recognition_results.xes", "uncertainty:classification_prob_windows_start-7_end-41_action_1")
    #n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_ptp_data_anonymize_recognition_results.xes",
                                                        # "uncertainty:classification_prob_windows_start-15_end-15_action_1")

    res_wrapper = dict()
    res_wrapper["n_acts"] = n_acts
    res_wrapper["n_repeats"] = n_repeats
    res_wrapper["n_steps"] = len(thres_values)
    res_wrapper["beta"] = beta
    res_wrapper["thres_values"] = thres_values
    res_wrapper["n_events"] = n_events
    res_wrapper["run_results"] = []
    res_wrapper["K"] = K
    for thres in tqdm(thres_values, desc="Varying uncertainty threshold (r)", leave=False):
        res_runs = dict()
        res_runs["thres"] = thres
        res_runs["runs"] = []
        for i in range(n_repeats):
            simulated_log = log_generation.simulate_randomly(n_events, thres, beta, n_acts, r_low=0.9, r_high=1.1)
            res = get_statistics(simulated_log, K)
            res["run"] = i
            res["timing"] = check_runtime(simulated_log, K)
            res_runs["runs"].append(res)
        res_wrapper["run_results"].append(res_runs)

    output_filename = os.path.join(BASE_OUT_DIR, out_dir, f"results.pkl")
    with open(output_filename, "wb") as file:
        pickle.dump(res_wrapper, file)

    return res_wrapper, full_out_dir


def sensitivity_analysis_k(n_events, n_acts, beta, thres, K_values, n_repeats=REPEAT):
    out_dir = datetime.now().strftime("%Y-%d-%m_%H-%M-%S") + f"_analyze_k_with_n_events_{n_events}_n_acts_{n_acts}_beta_{beta}_thres_{thres}"
    full_out_dir = os.path.join(BASE_OUT_DIR, out_dir)
    os.mkdir(full_out_dir)

    # n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_hr_data_anonymize_recognition_results.xes")
    # n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_ptp_data_anonymize_recognition_results.xes", "uncertainty:classification_prob_windows_start-7_end-41_action_1")
    #n_activities, ul = log_generation.load_uncertain_xes("data/uncertainty_ptp_data_anonymize_recognition_results.xes",
                                                        # "uncertainty:classification_prob_windows_start-15_end-15_action_1")

    res_wrapper = dict()
    res_wrapper["n_acts"] = n_acts
    res_wrapper["n_repeats"] = n_repeats
    res_wrapper["n_steps"] = len(K_values)
    res_wrapper["beta"] = beta
    res_wrapper["thres"] = thres
    res_wrapper["n_events"] = n_events
    res_wrapper["run_results"] = []
    res_wrapper["K_values"] = K_values
    for K in tqdm(K_values, desc="Varying K", leave=False):
        res_runs = dict()
        res_runs["K"] = K
        res_runs["runs"] = []
        for i in range(n_repeats):
            simulated_log = log_generation.simulate_randomly(n_events, thres, beta, n_acts, r_low=0.9, r_high=1.1)
            res = get_statistics(simulated_log, K)
            res["run"] = i
            res["timing"] = check_runtime(simulated_log, K)
            res_runs["runs"].append(res)
        res_wrapper["run_results"].append(res_runs)

    output_filename = os.path.join(BASE_OUT_DIR, out_dir, f"results.pkl")
    with open(output_filename, "wb") as file:
        pickle.dump(res_wrapper, file)

    return res_wrapper, full_out_dir


if __name__ == "__main__":
    data, n_acts_out_dir = sensitivity_analysis_n_acts(100, 0.3, 0.3, 10**4, 2, 10, n_repeats=REPEAT)
    statistics_table(data, "n_acts", save_path=n_acts_out_dir)

    n_events_vals = np.concatenate((np.arange(1, 50, 1), np.arange(50, 100, 10), np.arange(100, 1001, 50)))[::-1]
    data, n_events_out_dir = sensitivity_analysis_n_events(3, 0.3, 0.3, 10**4, n_events_vals, n_repeats=REPEAT)
    statistics_table(data, "n_events", save_path=n_events_out_dir)

    data, beta_out_dir = sensitivity_analysis_beta(100, 3, 0.3, 10**4, np.arange(0.01, 1.01, 0.01), n_repeats=REPEAT)
    statistics_table(data, "beta", save_path=beta_out_dir)

    k_vals = np.concatenate((np.arange(1, 10, 1), np.arange(10, 100, 5), np.arange(100, 1000, 25), np.arange(1000, 10001, 125)))
    data, k_out_dir = sensitivity_analysis_k(100, 3, 0.3, 0.3, k_vals, n_repeats=REPEAT)
    statistics_table(data, "K", save_path=k_out_dir)

    RUN_DIRS = {
        "beta": beta_out_dir,
        "n_events": n_events_out_dir,
        "n_acts": n_acts_out_dir,
        "K": k_out_dir
    }

    with open(os.path.join(BASE_OUT_DIR, f"sensitivity_analysis_results.pkl"), "wb") as file:
        pickle.dump(RUN_DIRS, file)
