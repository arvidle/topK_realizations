import numpy as np

from topK_realizations.algorithm import UncertainEventLog, realization_ranking, EventLog, Probability
from typing import Callable, Sequence, Iterable
from typing import Any
from collections import Counter, defaultdict
from scipy import stats
from functools import reduce
import math

type Activity = Any
type Trace = tuple[Activity, ...]
type SimpleEventLog = dict[Trace, int]


def simple_log_from_realization(realization: EventLog) -> SimpleEventLog:
    traces: dict[int, list[Activity]] = {}
    for _, (case, _, activity) in realization:
        if case not in traces.keys():
            traces[case] = []
        traces[case].append(activity)

    trace_counts: dict[tuple[Activity, ...], int] = {}

    for trace in traces.values():
        trace_tuple = tuple(trace)
        if trace_tuple not in trace_counts.keys():
            trace_counts[trace_tuple] = 0

        trace_counts[trace_tuple] += 1

    return trace_counts


def norm_entropy(values: np.ndarray, pre_normalize=True):
    if pre_normalize:
        values = values/sum(values)
    return stats.entropy(values)/math.log(len(values))


def norm_rel_entropy(values: np.ndarray, comp_values, pre_normalize=True):
    if pre_normalize:
        values = values/sum(values)
    return stats.entropy(values, comp_values)/math.log(len(values))


def check_split_point(probabilities: Sequence[Probability], thres: float, relative=True):
    prob_sum = sum(probabilities)
    target = prob_sum * thres
    running_sum = 0
    for i, probability in enumerate(probabilities):
        if running_sum >= target:
            if relative:
                return i / len(probabilities)
            else:
                return i
        else:
            running_sum += probability


def relative_split_point(probabilities: Sequence[Probability], thres: float, K: int):
    return check_split_point(probabilities, thres) / float(K)


def count_simple_event_logs(logs: Iterable[SimpleEventLog], sort_key) -> Counter:
    # Convert dicts to lists of tuples
    logs_as_list_of_tuples = [list(l.items()) for l in logs]
    # Sort each list of tuples by the 0-th element (keys)
    logs_as_sorted_list_of_tuples = [sorted(l, key=sort_key) for l in logs_as_list_of_tuples]
    # This gives us a list of list of tuples
    # To make this hashable, convert to tuple of tuples of tuples
    logs_as_tuple_of_tuples = tuple([tuple(l) for l in logs_as_sorted_list_of_tuples])
    # Then, apply a Counter
    return Counter(logs_as_tuple_of_tuples)


def sum_simple_event_log_probabilities(logs: list[SimpleEventLog], probabilities: list[float], sort_key) -> dict[SimpleEventLog, int]:
    res: dict[SimpleEventLog, int] = defaultdict(lambda: 0)
    # Convert dicts to lists of tuples
    logs_as_list_of_tuples = [list(l.items()) for l in logs]
    # Sort each list of tuples by the 0-th element (keys)
    logs_as_sorted_list_of_tuples = [sorted(l, key=sort_key) for l in logs_as_list_of_tuples]
    # This gives us a list of list of tuples
    # To make this hashable, convert to tuple of tuples of tuples
    logs_as_tuple_of_tuples = tuple([tuple(l) for l in logs_as_sorted_list_of_tuples])
    for probability, simple_log in zip(probabilities, logs_as_tuple_of_tuples):
        res[simple_log] += probability

    return dict(res)


def get_total_number_of_realizations(ul: UncertainEventLog):
    return reduce(lambda x, y: x * len(y), ul, 1)
