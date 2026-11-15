from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree


class BaseEstimator:
    def estimate(
        self,
        zone_arms: np.ndarray,
        zone_revenues: np.ndarray,
        arm_probs: np.ndarray,
        zone_neighbors: Optional[Dict[int, List[int]]] = None,
        fare_records: Optional[List[Tuple[int, float]]] = None,
        node_xy: Optional[np.ndarray] = None,
        node_zone: Optional[np.ndarray] = None,
        kdtree: Optional[KDTree] = None,
        interference_radius: float = 350.0,
        use_exposure_prob: bool = False,
        self_normalized: bool = False,
    ) -> Dict[int, float]:
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__


class SimpleAverage(BaseEstimator):
    def estimate(
        self,
        zone_arms,
        zone_revenues,
        arm_probs,
        zone_neighbors=None,
        fare_records=None,
        node_xy=None,
        node_zone=None,
        kdtree=None,
        interference_radius=350.0,
        use_exposure_prob=False,
        self_normalized=False,
    ):
        n_zones = len(zone_arms)
        n_arms = len(arm_probs)
        rewards = {}
        for arm in range(n_arms):
            mask = zone_arms == arm
            n_assigned = int(mask.sum())
            rewards[arm] = float(zone_revenues[mask].mean()) * n_zones if n_assigned > 0 else 0.0
        return rewards


def _parse_fare_record(record, node_zone, zone_arms):
    """Return (node_id, amount, arm, arm_prob) for old and new fare-record formats."""
    node_idx = None
    amount = None
    explicit_arm = None
    explicit_arm_prob = None

    if isinstance(record, dict):
        node_idx = record.get("node_id")
        amount = record.get("amount")
        explicit_arm = record.get("arm")
        explicit_arm_prob = record.get("arm_prob", record.get("assignment_prob"))
    elif isinstance(record, (list, tuple)):
        if len(record) >= 4 and isinstance(record[0], str):

            node_idx, amount, explicit_arm = record[1], record[2], record[3]
            if len(record) >= 7:
                explicit_arm_prob = record[6]
        elif len(record) >= 3:

            node_idx, amount, explicit_arm = record[0], record[1], record[2]
            if len(record) >= 6:
                explicit_arm_prob = record[5]
        elif len(record) >= 2:

            node_idx, amount = record[0], record[1]

    if node_idx is None or amount is None:
        return None

    node_idx = int(node_idx)
    zone_id = node_zone[node_idx] if node_idx < len(node_zone) else -1
    if explicit_arm is None:
        if zone_id < 0:
            return None
        arm = int(zone_arms[zone_id])
    else:
        arm = int(explicit_arm)

    arm_prob = None
    if explicit_arm_prob is not None:
        arm_prob = float(explicit_arm_prob)
        if not np.isfinite(arm_prob) or arm_prob <= 0:
            arm_prob = None

    return node_idx, float(amount), arm, arm_prob


def _build_pickup_arm_map(fare_records, node_zone, zone_arms):
    pickup_arm = {}
    for record in fare_records:
        parsed = _parse_fare_record(record, node_zone, zone_arms)
        if parsed is None:
            continue
        node_idx, _amount, arm, _arm_prob = parsed
        pickup_arm[node_idx] = int(arm)
    return pickup_arm


def _exposure_ok_passenger(node_idx, node_xy, kdtree, valid_idxs, pickup_arm, arm, radius):
    x_coord, y_coord = node_xy[node_idx]
    neighbor_kd_idxs = kdtree.query_ball_point([x_coord, y_coord], r=radius, p=np.inf)
    for kd_idx in neighbor_kd_idxs:
        neighbor_node_id = valid_idxs[kd_idx]
        if neighbor_node_id == node_idx:
            continue
        neighbor_arm = pickup_arm.get(neighbor_node_id)
        if neighbor_arm is not None and neighbor_arm != arm:
            return False
    return True


def _estimate_node_level(
    fare_records,
    node_xy,
    node_zone,
    zone_arms,
    arm_probs,
    beta,
    is_htix,
    interference_radius,
    use_exposure_prob,
    self_normalized,
):
    n_arms = len(arm_probs)
    n_zones = len(zone_arms)
    if len(fare_records) == 0:
        return {arm: 0.0 for arm in range(n_arms)}

    pickup_arm = _build_pickup_arm_map(fare_records, node_zone, zone_arms)
    pickup_node_ids = np.array(list(pickup_arm.keys()), dtype=int)
    if len(pickup_node_ids) == 0:
        return {arm: 0.0 for arm in range(n_arms)}

    pickup_xy = node_xy[pickup_node_ids]
    pickup_kdtree = KDTree(pickup_xy)

    rewards = {}
    for arm in range(n_arms):
        assignment_prob = float(arm_probs[arm])
        total = 0.0
        weight_sum = 0.0

        for record in fare_records:
            parsed = _parse_fare_record(record, node_zone, zone_arms)
            if parsed is None:
                continue
            node_idx, amount, record_arm, record_arm_prob = parsed
            zone_id = node_zone[node_idx] if node_idx < len(node_zone) else -1
            if zone_id < 0 or record_arm != arm:
                continue
            record_assignment_prob = (
                float(record_arm_prob) if record_arm_prob is not None else assignment_prob
            )
            if not _exposure_ok_passenger(
                node_idx,
                node_xy,
                pickup_kdtree,
                pickup_node_ids,
                pickup_arm,
                arm,
                interference_radius,
            ):
                continue

            if use_exposure_prob:
                x_coord, y_coord = node_xy[node_idx]
                neighbor_kd_idxs = pickup_kdtree.query_ball_point(
                    [x_coord, y_coord],
                    r=interference_radius,
                    p=np.inf,
                )
                distinct_zones = set()
                for kd_idx in neighbor_kd_idxs:
                    neighbor_node_id = pickup_node_ids[kd_idx]
                    neighbor_zone = node_zone[neighbor_node_id] if neighbor_node_id < len(node_zone) else -1
                    if neighbor_zone >= 0:
                        distinct_zones.add(int(neighbor_zone))
                exposure_prob = record_assignment_prob ** max(1, len(distinct_zones))
            else:
                exposure_prob = record_assignment_prob

            denom = exposure_prob + beta if is_htix else exposure_prob
            denom = max(float(denom), 1e-9)
            weight = 1.0 / denom
            total += float(amount) * weight
            weight_sum += weight

        if self_normalized:
            rewards[arm] = total / weight_sum if weight_sum > 0 else 0.0
        else:
            rewards[arm] = total / n_zones
    return rewards


class HT(BaseEstimator):
    def estimate(
        self,
        zone_arms,
        zone_revenues,
        arm_probs,
        zone_neighbors=None,
        fare_records=None,
        node_xy=None,
        node_zone=None,
        kdtree=None,
        interference_radius=350.0,
        use_exposure_prob=False,
        self_normalized=False,
    ):
        n_zones = len(zone_arms)
        n_arms = len(arm_probs)

        use_node_level = fare_records is not None and node_xy is not None and node_zone is not None
        if use_node_level:
            return _estimate_node_level(
                fare_records,
                node_xy,
                node_zone,
                zone_arms,
                arm_probs,
                beta=0.0,
                is_htix=False,
                interference_radius=interference_radius,
                use_exposure_prob=use_exposure_prob,
                self_normalized=self_normalized,
            )

        rewards = {}
        for arm in range(n_arms):
            assignment_prob = float(arm_probs[arm])
            if assignment_prob <= 0:
                rewards[arm] = 0.0
                continue
            total = 0.0
            n_used = 0
            for zone_id in range(n_zones):
                if zone_arms[zone_id] != arm:
                    continue
                if zone_neighbors is not None:
                    neighbors = zone_neighbors.get(zone_id, [])
                    if not all(zone_arms[neighbor] == arm for neighbor in neighbors):
                        continue
                total += float(zone_revenues[zone_id]) / assignment_prob
                n_used += 1
            rewards[arm] = total / n_zones if n_used > 0 else 0.0
        return rewards


class HT_IX(BaseEstimator):
    def __init__(self, beta: float = 0.1):
        self.beta = float(beta)

    def estimate(
        self,
        zone_arms,
        zone_revenues,
        arm_probs,
        zone_neighbors=None,
        fare_records=None,
        node_xy=None,
        node_zone=None,
        kdtree=None,
        interference_radius=350.0,
        use_exposure_prob=False,
        self_normalized=False,
    ):
        n_zones = len(zone_arms)
        n_arms = len(arm_probs)

        use_node_level = fare_records is not None and node_xy is not None and node_zone is not None
        if use_node_level:
            return _estimate_node_level(
                fare_records,
                node_xy,
                node_zone,
                zone_arms,
                arm_probs,
                beta=self.beta,
                is_htix=True,
                interference_radius=interference_radius,
                use_exposure_prob=use_exposure_prob,
                self_normalized=self_normalized,
            )

        rewards = {}
        for arm in range(n_arms):
            assignment_prob = float(arm_probs[arm])
            total = 0.0
            n_used = 0
            for zone_id in range(n_zones):
                if zone_arms[zone_id] != arm:
                    continue
                if zone_neighbors is not None:
                    neighbors = zone_neighbors.get(zone_id, [])
                    if not all(zone_arms[neighbor] == arm for neighbor in neighbors):
                        continue
                total += float(zone_revenues[zone_id]) / (assignment_prob + self.beta)
                n_used += 1
            rewards[arm] = total / n_zones if n_used > 0 else 0.0
        return rewards

    def name(self):
        return f"HT_IX(beta={self.beta})"


def get_estimator(name: str, **kwargs) -> BaseEstimator:
    estimators = {
        "SimpleAverage": SimpleAverage,
        "HT": HT,
        "HT_IX": HT_IX,
    }
    if name not in estimators:
        raise ValueError(f"Unknown estimator: {name}. Available: {list(estimators.keys())}")
    return estimators[name](**kwargs)
