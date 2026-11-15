import json
import logging

import numpy as np

from src.demand.TravelerModels import IndividualConstraintRequest, _stable_uniform_01
from src.misc.globals import (
    G_NR_OPERATORS,
    G_OFFER_DRIVE,
    G_OFFER_FARE,
    G_OFFER_WAIT,
    G_RANDOM_SEED,
    G_RQ_MAX_FARE,
)


LOG = logging.getLogger(__name__)

FLOOR_KEY = "rq_prob_accept_floor"
CEILING_KEY = "rq_prob_accept_ceiling"
SEED_KEY = "rq_prob_accept_seed"
WAIT_PENALTY_KEY = "rq_prob_accept_wait_penalty_per_min"
WAIT_REFERENCE_KEY = "rq_prob_accept_wait_reference_min"
SCHEDULE_KEY = "rq_logit_coefficient_schedule"


INPUT_PARAMETERS_TimeVaryingLogitPriceSensitiveIndividualConstraintRequest = {
    "doc": "Phase-dependent binary-logit passenger acceptance.",
    "inherit": "IndividualConstraintRequest",
    "input_parameters_mandatory": [G_RQ_MAX_FARE],
    "input_parameters_optional": [
        FLOOR_KEY,
        CEILING_KEY,
        SEED_KEY,
        WAIT_PENALTY_KEY,
        WAIT_REFERENCE_KEY,
        SCHEDULE_KEY,
    ],
    "mandatory_modules": [],
    "optional_modules": [],
}


def _parse_schedule(value):
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{SCHEDULE_KEY} must be a list of mappings.")
    schedule = []
    for entry in value:
        try:
            start = float(entry["start_time"])
            end = float(entry["end_time"])
            intercept = float(entry["intercept"])
            sensitivity = float(entry["price_sensitivity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Each schedule entry needs start_time, end_time, intercept, and price_sensitivity.") from exc
        if not (
            np.isfinite(start)
            and np.isfinite(end)
            and np.isfinite(intercept)
            and np.isfinite(sensitivity)
            and 0.0 <= start <= 86400.0
            and 0.0 <= end <= 86400.0
            and start != end
            and sensitivity >= 0.0
        ):
            raise ValueError("Invalid binary-logit schedule entry.")
        schedule.append((start, end, intercept, sensitivity))
    return tuple(schedule)


def _coefficients_at_time(schedule, simulation_time):
    if not schedule:
        return 0.0, 1.0
    time_of_day = float(simulation_time) % 86400.0
    for start, end, intercept, sensitivity in schedule:
        active = start <= time_of_day < end if start < end else time_of_day >= start or time_of_day < end
        if active:
            return intercept, sensitivity
    raise ValueError(f"{SCHEDULE_KEY} does not cover {time_of_day:.3f} seconds.")


class TimeVaryingLogitPriceSensitiveIndividualConstraintRequest(IndividualConstraintRequest):
    type = "TimeVaryingLogitPriceSensitiveIndividualConstraintRequest"

    def __init__(self, rq_row, routing_engine, simulation_time_step, scenario_parameters):
        super().__init__(rq_row, routing_engine, simulation_time_step, scenario_parameters)
        self.max_fare = float(rq_row[G_RQ_MAX_FARE])
        self.schedule = _parse_schedule(scenario_parameters.get(SCHEDULE_KEY))

    def _acceptance_probability(self, offered_fare, parameters, simulation_time, offered_wait_time):
        if offered_fare is None:
            return 1.0
        intercept, sensitivity = _coefficients_at_time(self.schedule, simulation_time)
        fare_ratio = max(float(offered_fare), 0.0) / max(self.max_fare, 1e-9)
        wait_minutes = 0.0 if offered_wait_time is None else max(float(offered_wait_time), 0.0) / 60.0
        wait_penalty = float(parameters.get(WAIT_PENALTY_KEY, 0.0))
        wait_reference = float(parameters.get(WAIT_REFERENCE_KEY, 0.0))
        utility = np.clip(
            intercept - sensitivity * fare_ratio - wait_penalty * (wait_minutes - wait_reference),
            -60.0,
            60.0,
        )
        probability = 1.0 / (1.0 + np.exp(-utility))
        floor = min(max(float(parameters.get(FLOOR_KEY, 0.0)), 0.0), 1.0)
        ceiling = min(max(float(parameters.get(CEILING_KEY, 1.0)), floor), 1.0)
        return floor + (ceiling - floor) * probability

    def choose_offer(self, scenario_parameters, simulation_time):
        declined = [offer_id for offer_id, offer in self.offer.items() if offer.service_declined()]
        if len(declined) == scenario_parameters[G_NR_OPERATORS]:
            return -1
        operators = sorted(offer_id for offer_id in self.offer if offer_id >= 0)
        if not operators:
            return None
        seed = int(scenario_parameters.get(SEED_KEY, scenario_parameters.get(G_RANDOM_SEED, 0)))
        for operator_id in operators:
            offer = self.offer[operator_id]
            offered_fare = offer.get(G_OFFER_FARE)
            probability = self._acceptance_probability(
                offered_fare,
                scenario_parameters,
                simulation_time,
                offer.get(G_OFFER_WAIT),
            )
            if _stable_uniform_01(self.rid, simulation_time, seed) > probability:
                return -1
            pickup_time = self.rq_time + offer[G_OFFER_WAIT]
            if pickup_time < self.earliest_start_time or pickup_time > self.latest_start_time:
                return -1
            if self.max_trip_time and offer[G_OFFER_DRIVE] > self.max_trip_time:
                return -1
            self.fare = offer.get(G_OFFER_FARE, 0)
            return operator_id
        return -1
