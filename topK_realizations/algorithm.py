from typing import Any, Union
from collections.abc import Iterator
import functools
from heapq import heappop, heappush

type Probability = float
type EventRealizationProbability = Probability
type Event = Any
type EventID = int
type Choice = tuple[EventRealizationProbability, Event]
type UncertainEvent = list[Choice]
type UncertainEventLog = list[UncertainEvent]
type RealizationIndex = int
type ForcedRealization = bool

# Represents a substitution operation that transforms a realization into another realization that is different in
# exactly one element
# 1. Event ID
# 2. Original realization index
# 3. Substitute realization index
type Substitution = tuple[EventID, RealizationIndex, RealizationIndex]

# This type is special - it refers to the indices of the choices in the uncertain log
# To get the actual choices associated with this realization, use function get_realization_choices
# The bool indicates whether this event is forced in the corresponding partition where this is the top-1 solution
type LogRealization = list[tuple[RealizationIndex, ForcedRealization]]

# The restricted second best realization always refers to a restricted subspace (consult the paper for details)
# It has three parts:
# 1. The actual log realization
# 2. The probability of this realization (as calculated by applying the probability ratio relative to a best solution)
# 3. The substitution operation that was used to derive this realization. Note that the first realization index
#    represents exactly the singular element of D_1 \ D_1^2 and the second realization index the one element that is
#    in D_1^2 \ D_1, so this can be used for the partitioning step. (And in principle, D_1 could be reconstructed.)
type RestrictedSecondBestRealization = tuple[LogRealization, Probability, Probability, Substitution]

type EventLog = list[Event]


class MaxHeap:
    def __init__(self):
        self.heap = []

    def push(self, element):
        heappush(self.heap, (-1 * element[1], (element[0], element[2], element[3])))

    def pop(self):
        element = heappop(self.heap)
        return element[1][0], -1 * element[0], element[1][1], element[1][2]

    def is_empty(self):
        return len(self.heap) == 0


# For the code down below, we assume that the list of choices that constitutes an uncertain event is sorted
# with respect to the total order implied by <= on the probabilities of the different choices.
def get_realization_choices(log_realization: LogRealization, uncertain_log: UncertainEventLog) -> EventLog:
    return [uncertain_log[event_id][realization_idx] for event_id, (realization_idx, _) in enumerate(log_realization)]


def get_choices_from_indices(log_realization: list[RealizationIndex], uncertain_log: UncertainEventLog) -> EventLog:
    return [uncertain_log[event_id][realization_idx] for event_id, realization_idx in enumerate(log_realization)]


def event_realization_probability(choice: Choice) -> Probability:
    return choice[0]


def log_realization_probability(log_realization: LogRealization, uncertain_log: UncertainEventLog) -> Probability:
    return functools.reduce(lambda a, b: a * b,
                            [event_realization_probability(choice) for choice in get_realization_choices(log_realization, uncertain_log)], 1)


def best_realization(uncertain_log: UncertainEventLog) -> tuple[Probability, LogRealization]:
    top_1 = [(0, False) for _ in range(len(uncertain_log))]
    p_1 = log_realization_probability(top_1, uncertain_log)

    return p_1, top_1


# We also return the status here because at some point there is no second best solution anymore (i.e., when we are
# searching in a partition of size 1)
def restricted_second_best_substitution(uncertain_log: UncertainEventLog, best_realization: LogRealization) -> tuple[bool, Probability, Substitution]:
    best_substitute = None
    best_substitute_ratio = 0.0

    for i in range(len(uncertain_log)):
        if (best_realization[i][1]) or (len(uncertain_log[i]) <= best_realization[i][0] + 1):
            continue
        else:
            best_idx = best_realization[i][0]
            best_probability = uncertain_log[i][best_idx][0]
            next_best_idx = best_idx + 1
            next_best_prob = uncertain_log[i][next_best_idx][0]

            substitute_ratio = next_best_prob / best_probability
            if substitute_ratio >= best_substitute_ratio:
                best_substitute = (i, best_idx, next_best_idx)
                best_substitute_ratio = substitute_ratio

    if best_substitute is not None:
        return True, best_substitute_ratio, best_substitute
    else:
        return False, 1.0, (0, 0, 0)


# This could also return the probability ratio (or the full probability of the new realization)
def substitute_in_realization(base_realization: LogRealization, substitution: Substitution) -> LogRealization:
    new_realization = base_realization.copy()
    new_realization[substitution[0]] = (substitution[2], new_realization[substitution[0]][1])

    return new_realization


def restricted_second_best_realization(uncertain_log: UncertainEventLog, base_realization: LogRealization, base_probability: Probability) -> tuple[bool, RestrictedSecondBestRealization]:
    success, probability_ratio, substitution = restricted_second_best_substitution(uncertain_log, base_realization)
    if success:
        return success, (substitute_in_realization(base_realization, substitution), base_probability * probability_ratio, probability_ratio, substitution)
    else:
        return success, (base_realization, base_probability, 1.0, substitution)


def realization_ranking(uncertain_log: UncertainEventLog, yield_realization: bool = False) -> Union[Iterator[Probability, EventLog], Iterator[Probability, EventLog, LogRealization]]:
    realization_heap = MaxHeap()
    probability_top_1, top_1 = best_realization(uncertain_log)
    p_1 = probability_top_1
    if yield_realization:
        yield p_1, get_realization_choices(top_1, uncertain_log), top_1
    else:
        yield p_1, get_realization_choices(top_1, uncertain_log)

    next_best_exists, (top_1_2, p_1_2, rho_p, substitution) = restricted_second_best_realization(uncertain_log, top_1, p_1)
    if next_best_exists:
        realization_heap.push((top_1_2, p_1_2, rho_p, substitution))
    else:
        return

    while not realization_heap.is_empty():
        next_best_realization, next_best_probability, next_best_probability_ratio, current_substitution = realization_heap.pop()
        base_probability = next_best_probability / next_best_probability_ratio
        if yield_realization:
            yield next_best_probability, get_realization_choices(next_best_realization, uncertain_log), next_best_realization
        else:
            yield next_best_probability, get_realization_choices(next_best_realization, uncertain_log)

        sub_event_ID, sub_original_idx, sub_new_idx = current_substitution
        # Here, do the splitting, add both R-2-best to heap, and so on
        # For the left side, we force the one element from D_1 \ D_1^2 by setting the force boolean of the
        # corresponding event realization to True
        forced_top_1 = next_best_realization.copy()
        assert not forced_top_1[sub_event_ID][1]
        forced_top_1[sub_event_ID] = (sub_original_idx, True)
        forced_success, forced_second_best = restricted_second_best_realization(uncertain_log, forced_top_1, base_probability)
        if forced_success:
            realization_heap.push(forced_second_best)

        # For the right side, we exclude the split element by advancing the start (realization) index of the log
        # realization at the event ID from the substitution past the split element (so +1)
        excluded_top_1 = next_best_realization.copy()
        assert not excluded_top_1[sub_event_ID][1]
        excluded_top_1[sub_event_ID] = (sub_original_idx + 1, False)
        excluded_success, excluded_second_best = restricted_second_best_realization(uncertain_log, excluded_top_1, next_best_probability)
        if excluded_success:
            realization_heap.push(excluded_second_best)

    return


def realization_ranking_baseline(uncertain_log: UncertainEventLog, yield_realization: bool = False) -> Union[Iterator[Probability, EventLog], Iterator[Probability, EventLog, LogRealization]]:
    # Calculate all realizations and their probabilities
    current_realizations = [(0.0, [])]
    temp_realizations = []
    for uncertain_event in uncertain_log:
        for probability, realization in current_realizations:
            for event_realization_id, (event_probability, _) in enumerate(uncertain_event):
                next_probability = probability * event_probability
                next_realization = realization + [event_realization_id]
                temp_realizations.append((next_probability, next_realization))
        current_realizations = temp_realizations
        temp_realizations = []

    # Return an iterator
    for probability, realization in sorted(current_realizations, key=lambda x: x[0], reverse=True):
        log = []

        if yield_realization:
            yield probability, log, realization
        else:
            yield probability, log

        return
