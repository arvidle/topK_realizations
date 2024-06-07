import numpy as np
from topK_realizations.algorithm import UncertainEvent, EventLog, UncertainEventLog
import pm4py
import pandas as pd
from pm4py.objects.petri_net.obj import PetriNet, Marking
from typing import Any, Sequence
from math import log, floor
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
from itertools import accumulate
from functools import reduce

type PM4PYActivity = Any
type PM4PYTimestamp = Any
type PM4PYCase = Any
type PM4PYEvent = tuple[PM4PYActivity, PM4PYTimestamp, PM4PYCase]
type PM4PYEventLog = list[PM4PYEvent]



def load_xes_event_log(filename: str) -> pd.DataFrame:
    return pm4py.read_xes(filename)


def load_csv_event_log(filename: str) -> pd.DataFrame:
    log = pd.read_csv(filename)
    return log


def load_petri_net(pnml_file: str) -> tuple[PetriNet, Marking, Marking]:
    net, im, fm = pm4py.read_pnml(pnml_file)
    return net, im, fm


def gen_log_from_model(model: PetriNet, im: Marking, fm: Marking, n_cases: int) -> pd.DataFrame:
    log = pm4py.algo.simulation.playout.petri_net.algorithm.apply(model, im, fm, parameters={"noTraces": n_cases})
    return pm4py.convert_to_dataframe(log)


def pm4py_log_to_event_log(log: pd.DataFrame) -> EventLog:
    # pm4py event log has 3 main columns:
    # concept:name for the activity name/identifier (e.g., as a str)
    # time:timestamp for the timestamp (probably in some datetime format??)
    # case:concept:name for the case identifier (e.g., as an int)
    #
    # Our EventLog type is just a list of Events, which themselves can be anything
    res: PM4PYEventLog = []

    for idx, event in log.iterrows():
        activity: PM4PYActivity = event["concept:name"]
        timestamp: PM4PYTimestamp = event["time:timestamp"]
        case: PM4PYCase = event["case:concept:name"]

        transformed_event: PM4PYEvent = (activity, timestamp, case)
        res.append(transformed_event)

    return res


def event_log_to_pm4py_log(log: EventLog) -> pd.DataFrame:
    _, events = zip(*log)
    res: pd.DataFrame = pd.DataFrame(list(events), columns=["case:concept:name", "time:timestamp", "concept:name"])
    res["case:concept:name"] = res["case:concept:name"].astype(str)
    res["concept:name"] = res["concept:name"].astype(str)
    return res


def load_uncertain_xes(path: str, uncertain_activities_key: str = "uncertainty:name") -> tuple[int, UncertainEventLog]:
    event_log: pd.DataFrame = pm4py.read_xes(path)

    res: UncertainEventLog = []
    for idx, event in event_log.iterrows():
        top_activity: PM4PYActivity = event["concept:name"]
        timestamp: PM4PYTimestamp = event["time:timestamp"]
        case: PM4PYCase = floor(float(event["case:concept:name"]))

        if type(event[uncertain_activities_key]) == str:
            activity_realizations = [(event["concept:name"], 1.0)]
        else:
            activity_realizations = sorted(event[uncertain_activities_key]["children"].items(), key=lambda x: x[1], reverse=True)
        current_choices: UncertainEvent = [(probability, (case, timestamp, activity)) for activity, probability in activity_realizations]

        res.append(current_choices)

    return len(event_log["concept:name"].unique()), res


def f(c):
    return lambda x: 2**(-1 * (2**x - 1) * (2**c - 1))


def sample_evenly(g, n_samples, low=0, high=1, normalize=True):
    xs = np.arange(low, high, (high - low) / n_samples)
    res = g(xs)

    if normalize:
        res = res/sum(res)

    return res


def sample_randomly(g, n_samples, low=0, high=1.0, normalize=True, eps=1e-5):
    xs = np.sort(np.random.uniform(low, high, n_samples))
    res = g(xs)

    res[res < eps] = 0

    if normalize:
        if sum(res) < eps:
            res = np.zeros(len(res))
            res[0] = 1.0
        res = res/sum(res)

    return res


def uncertainty_measure(ps):
    return stats.entropy(ps, np.ones(len(ps))/len(ps))/log(len(ps))


def gen_vec(threshold: float, pick_n: int, r_low: float = 0, r_high: float = 1, normalize: bool = True) -> list[float]:
    values: np.ndarray = np.zeros(pick_n)
    values[0] = 1
    res: list[float] = list((accumulate(values, lambda x, _: x * np.random.uniform(low=r_low, high=r_high) * threshold)))
    res_arr: np.ndarray = np.array(res)
    res_arr = res_arr/sum(res_arr) if normalize else res_arr
    return [p for p in res_arr]


def get_unique_case_ids(ul: UncertainEventLog) -> list[PM4PYCase]:
    return list(set(reduce(lambda cs, e: cs + [c for (_, (c, _, _)) in e], ul, [])))


def substitute_case_ids(ut: UncertainEventLog, case_id: int) -> UncertainEventLog:
    return [[(p, (case_id, t, a)) for (p, (_, t, a)) in e] for e in ut]


def separate_cases(ul: UncertainEventLog) -> list[UncertainEventLog]:
    cases: dict[int, UncertainEventLog] = {}
    for event in ul:
        _, (c, _, _) = event[0]
        if c not in cases.keys():
            cases[c] = []
        cases[c].append(event)

    return list(cases.values())


def gen_fully_random_prob_vector(activities: Sequence[Any], n_activities=3) -> tuple[list[Any], Sequence[float]]:
    activities = random.sample(activities, n_activities)
    vec = np.random.uniform(size=n_activities)
    norm_vec = vec / sum(vec)
    return activities, norm_vec


def simulate_randomly(n_events: int, uncertainty_threshold: float, skewness_factor: float, n_acts: int, activities: Sequence[Any] = None, r_low=0.9, r_high=1.1) -> UncertainEventLog:
    if activities is None:
        activities = [str(i) for i in range(n_acts)]

    res: UncertainEventLog = []
    uncertain_idxs = set(random.sample(range(n_events), round(uncertainty_threshold * n_events)))

    for i in range(n_events):
        if i in uncertain_idxs:
            event_activities = random.sample(activities, n_acts)
            event_probabilities = gen_vec(skewness_factor, n_acts, r_low, r_high)
            current_event: UncertainEvent = [(prob, act) for (prob, act) in zip(event_probabilities, event_activities)]
        else:
            current_activity = random.choice(activities)
            current_event: UncertainEvent = [(1.0, current_activity)]

        res.append(current_event)

    return res


def simulate_from_log(ul: UncertainEventLog, n_cases: int, choice_prob_threshold: float, r_low: float = 0.5, r_high: float = 1) -> UncertainEventLog:
    cases = separate_cases(ul)
    if n_cases < len(cases):
        cases_picks = np.random.choice(len(cases), n_cases, replace=False)
        res_cases = [cases[idx] for idx in cases_picks]
    elif n_cases > len(cases):
        additional_cases_picks = np.random.choice(len(cases), n_cases, replace=True)
        additional_cases = [cases[idx] for idx in additional_cases_picks]
        res_cases = cases + additional_cases
    else:
        assert n_cases == len(cases)
        res_cases = cases

    res_log: UncertainEventLog = []

    for new_case_id, case in enumerate(res_cases):
        new_case: UncertainEventLog = substitute_case_ids(case, new_case_id)
        # We assume here that the choices are sorted by descending realization probability
        # This should be handled at import

        for uncertain_event in new_case:
            choice_probability_distribution = gen_vec(choice_prob_threshold, len(new_case), r_low, r_high)
            new_uncertain_event = ([(new_probability, (c, t, a)) for (new_probability, (_, (c, t, a))) in zip(choice_probability_distribution, uncertain_event)])
            res_log.append(sorted(new_uncertain_event, key=lambda x: x[0], reverse=True))

    return res_log


def simulate_growing_log(ul: UncertainEventLog, n_cases_init: int, n_cases_step: int, choice_prob_threshold: float, r_low: float = 0.5, r_high: float = 1) -> UncertainEventLog:
    # TODO: Enable this to randomly add more and more cases (by making it a generator?)
    # Then, the event logs using smaller runs are guaranteed to use a subset of the longer logs that are used later
    cases = separate_cases(ul)
    if n_cases_init < len(cases):
        cases_picks = np.random.choice(len(cases), n_cases_init, replace=False)
        res_cases = [cases[idx] for idx in cases_picks]
    elif n_cases_init > len(cases):
        additional_cases_picks = np.random.choice(len(cases), n_cases_init, replace=True)
        additional_cases = [cases[idx] for idx in additional_cases_picks]
        res_cases = cases + additional_cases
    else:
        assert n_cases_init == len(cases)
        res_cases = cases

    res_log: UncertainEventLog = []

    for new_case_id, case in enumerate(res_cases):
        new_case: UncertainEventLog = substitute_case_ids(case, new_case_id)
        # We assume here that the choices are sorted by descending realization probability
        # This should be handled at import

        for uncertain_event in new_case:
            choice_probability_distribution = gen_vec(choice_prob_threshold, len(new_case), r_low, r_high)
            new_uncertain_event = ([(new_probability, (c, t, a)) for (new_probability, (_, (c, t, a))) in
                                    zip(choice_probability_distribution, uncertain_event)])
            res_log.append(sorted(new_uncertain_event, key=lambda x: x[0], reverse=True))

    yield res_log

    while True:
        step_cases_picks = np.random.choice(len(cases), n_cases_step, replace=True)
        step_cases = [cases[idx] for idx in step_cases_picks]

        for new_case_id, case in enumerate(step_cases):
            new_case: UncertainEventLog = substitute_case_ids(case, new_case_id)
            # We assume here that the choices are sorted by descending realization probability
            # This should be handled at import

            for uncertain_event in new_case:
                choice_probability_distribution = gen_vec(choice_prob_threshold, len(new_case), r_low, r_high)
                new_uncertain_event = ([(new_probability, (c, t, a)) for (new_probability, (_, (c, t, a))) in
                                        zip(choice_probability_distribution, uncertain_event)])
                res_log.append(sorted(new_uncertain_event, key=lambda x: x[0], reverse=True))

        yield res_log
