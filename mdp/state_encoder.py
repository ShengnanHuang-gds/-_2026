"""
Convert MDPState into a fixed-length float vector for neural RL.

Two encoders:

1) StateEncoder ("raw") — inventory / order-up-to ratios (default 94-d)
2) RiskFeatureEncoder ("risk") — DOS / shortage / disruption aggregates (~33-d)
"""
from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np

from mdp.env import MDPState
from models.central_warehouse import CentralWarehouse
from models.forward_warehouse import ForwardWarehouse


PIPELINE_BUCKETS = (1, 2, 3, 4)  # last bucket = remaining_lead_time >= 4

VALID_ENCODER_TYPES = frozenset({"raw", "risk"})
EncoderLike = Union["StateEncoder", "RiskFeatureEncoder"]


class StateEncoder:
    """Deterministic, normalized state featurizer."""

    def __init__(
        self,
        central_order_up_to: Sequence[int],
        forward_order_up_to: Sequence[Sequence[int]],
        evaluation_days: int,
        num_products: int = 5,
    ) -> None:
        if evaluation_days <= 0:
            raise ValueError("evaluation_days must be positive")
        if len(central_order_up_to) != num_products:
            raise ValueError("central_order_up_to length must equal num_products")
        if any(level <= 0 for level in central_order_up_to):
            raise ValueError("central_order_up_to levels must be positive")

        self.num_products = num_products
        self.num_forward_warehouses = len(forward_order_up_to)
        self.evaluation_days = int(evaluation_days)
        self.central_order_up_to = [float(x) for x in central_order_up_to]
        self.forward_order_up_to = [
            [float(x) for x in row] for row in forward_order_up_to
        ]
        for row in self.forward_order_up_to:
            if len(row) != num_products:
                raise ValueError("each FW order-up-to row must have num_products")
            if any(level <= 0 for level in row):
                raise ValueError("forward_order_up_to levels must be positive")

        self._dim = (
            num_products  # CW
            + self.num_forward_warehouses * num_products  # FW
            + len(PIPELINE_BUCKETS) * num_products  # pipeline
            + num_products  # backlog
            + num_products  # waiting
            + 4  # A one-hot
            + 4  # U one-hot
            + 1  # day progress
        )

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, state: MDPState) -> np.ndarray:
        """Return float32 vector of shape (dim,)."""
        features: List[float] = []

        # CW inventory
        for k, level in enumerate(state.central_inventory):
            features.append(level / self.central_order_up_to[k])

        # FW inventory (ordered by encoder's FW index 0..N-1)
        if len(state.forward_inventory) != self.num_forward_warehouses:
            raise ValueError(
                f"expected {self.num_forward_warehouses} FW inventories, "
                f"got {len(state.forward_inventory)}"
            )
        for i, fw_inv in enumerate(state.forward_inventory):
            for k, level in enumerate(fw_inv):
                features.append(level / self.forward_order_up_to[i][k])

        # Pipeline buckets per product
        bucket_totals = [
            [0.0 for _ in range(self.num_products)]
            for _ in PIPELINE_BUCKETS
        ]
        for quantities, remaining_lt in state.supplier_pipeline:
            if remaining_lt <= 0:
                continue
            if remaining_lt == 1:
                bucket_index = 0
            elif remaining_lt == 2:
                bucket_index = 1
            elif remaining_lt == 3:
                bucket_index = 2
            else:
                bucket_index = 3
            for k, qty in enumerate(quantities):
                bucket_totals[bucket_index][k] += qty
        for bucket in bucket_totals:
            for k, qty in enumerate(bucket):
                features.append(qty / self.central_order_up_to[k])

        # Upstream buffers
        for k, qty in enumerate(state.supplier_backlog):
            features.append(qty / self.central_order_up_to[k])
        for k, qty in enumerate(state.transport_waiting):
            features.append(qty / self.central_order_up_to[k])

        # Disruption one-hots
        a_oh = [0.0, 0.0, 0.0, 0.0]
        u_oh = [0.0, 0.0, 0.0, 0.0]
        a = int(state.supplier_availability)
        u = int(state.transport_state)
        if not (0 <= a <= 3 and 0 <= u <= 3):
            raise ValueError(f"invalid disruption indices A={a}, U={u}")
        a_oh[a] = 1.0
        u_oh[u] = 1.0
        features.extend(a_oh)
        features.extend(u_oh)

        # Day progress in [0, 1]
        features.append(min(1.0, max(0.0, state.day / self.evaluation_days)))

        vector = np.asarray(features, dtype=np.float32)
        if vector.shape != (self._dim,):
            raise RuntimeError(
                f"encoder produced shape {vector.shape}, expected ({self._dim},)"
            )
        return vector


class RiskFeatureEncoder:
    """
    Compact decision-oriented features (default 10 FW x 5 SKU → 33-d):

      A one-hot (4) + U one-hot (4)
      is_severe, is_mild, is_u3                 (3)
      CW total DOS, CW min-product DOS          (2)
      FW mean DOS, FW min DOS                   (2)
      FW stockout ratio, FW low-DOS (<1) ratio  (2)
      per-SKU: pipe_1d/μ, pipe_3d/μ, (b+w)/μ   (15)
      day / evaluation_days                     (1)
      --------------------------------------------
      total                                     33
    """

    LOW_DOS_THRESHOLD = 1.0

    def __init__(
        self,
        fw_daily_demand_means: Sequence[Sequence[float]],
        evaluation_days: int,
        num_products: int = 5,
        eps: float = 1e-6,
    ) -> None:
        if evaluation_days <= 0:
            raise ValueError("evaluation_days must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not fw_daily_demand_means:
            raise ValueError("fw_daily_demand_means must be non-empty")

        self.num_products = num_products
        self.num_forward_warehouses = len(fw_daily_demand_means)
        self.evaluation_days = int(evaluation_days)
        self.eps = float(eps)
        self.fw_demand = [
            [float(x) for x in row] for row in fw_daily_demand_means
        ]
        for row in self.fw_demand:
            if len(row) != num_products:
                raise ValueError("each FW demand row must have num_products")
            if any(x < 0 for x in row):
                raise ValueError("daily demand means must be non-negative")

        self.cw_demand = [
            sum(self.fw_demand[i][k] for i in range(self.num_forward_warehouses))
            for k in range(num_products)
        ]
        self.cw_demand_total = float(sum(self.cw_demand))
        self._dim = 4 + 4 + 3 + 2 + 2 + 2 + 3 * num_products + 1

    @property
    def dim(self) -> int:
        return self._dim

    def _pipeline_totals(self, state: MDPState) -> tuple:
        """Return (pipe_1d[k], pipe_3d[k]) lists over products."""
        pipe_1d = [0.0] * self.num_products
        pipe_3d = [0.0] * self.num_products
        for quantities, remaining_lt in state.supplier_pipeline:
            if remaining_lt <= 0:
                continue
            for k, qty in enumerate(quantities):
                q = float(qty)
                if remaining_lt == 1:
                    pipe_1d[k] += q
                if remaining_lt <= 3:
                    pipe_3d[k] += q
        return pipe_1d, pipe_3d

    def encode(self, state: MDPState) -> np.ndarray:
        if len(state.forward_inventory) != self.num_forward_warehouses:
            raise ValueError(
                f"expected {self.num_forward_warehouses} FW inventories, "
                f"got {len(state.forward_inventory)}"
            )
        features: List[float] = []

        a = int(state.supplier_availability)
        u = int(state.transport_state)
        if not (0 <= a <= 3 and 0 <= u <= 3):
            raise ValueError(f"invalid disruption indices A={a}, U={u}")

        a_oh = [0.0, 0.0, 0.0, 0.0]
        u_oh = [0.0, 0.0, 0.0, 0.0]
        a_oh[a] = 1.0
        u_oh[u] = 1.0
        features.extend(a_oh)
        features.extend(u_oh)

        is_severe = 1.0 if (a >= 2 or u >= 2) else 0.0
        is_normal = 1.0 if (a == 0 and u == 0) else 0.0
        is_mild = 1.0 if (is_severe < 0.5 and is_normal < 0.5) else 0.0
        is_u3 = 1.0 if u == 3 else 0.0
        features.extend([is_severe, is_mild, is_u3])

        # CW DOS
        cw_inv = [float(x) for x in state.central_inventory]
        cw_total_dos = sum(cw_inv) / (self.cw_demand_total + self.eps)
        cw_min_dos = min(
            cw_inv[k] / (self.cw_demand[k] + self.eps)
            for k in range(self.num_products)
        )
        features.extend([cw_total_dos, cw_min_dos])

        # FW DOS stats over all (i, k)
        dos_vals: List[float] = []
        stockouts = 0
        low_dos = 0
        n_cells = self.num_forward_warehouses * self.num_products
        for i, fw_inv in enumerate(state.forward_inventory):
            for k, level in enumerate(fw_inv):
                mu = self.fw_demand[i][k]
                dos = float(level) / (mu + self.eps)
                dos_vals.append(dos)
                if level <= 0:
                    stockouts += 1
                if dos < self.LOW_DOS_THRESHOLD:
                    low_dos += 1
        features.append(float(sum(dos_vals) / n_cells))
        features.append(float(min(dos_vals)))
        features.append(stockouts / float(n_cells))
        features.append(low_dos / float(n_cells))

        pipe_1d, pipe_3d = self._pipeline_totals(state)
        backlog = [float(x) for x in state.supplier_backlog]
        waiting = [float(x) for x in state.transport_waiting]
        for k in range(self.num_products):
            mu = self.cw_demand[k] + self.eps
            features.append(pipe_1d[k] / mu)
            features.append(pipe_3d[k] / mu)
            features.append((backlog[k] + waiting[k]) / mu)

        features.append(min(1.0, max(0.0, state.day / self.evaluation_days)))

        vector = np.asarray(features, dtype=np.float32)
        if vector.shape != (self._dim,):
            raise RuntimeError(
                f"risk encoder produced shape {vector.shape}, expected ({self._dim},)"
            )
        return vector


def build_state_encoder_from_warehouses(
    central_warehouse: CentralWarehouse,
    forward_warehouses: Sequence[ForwardWarehouse],
    evaluation_days: int,
    *,
    encoder_type: str = "raw",
) -> EncoderLike:
    """Build encoder using warehouse baselines / demand means."""
    et = str(encoder_type).lower()
    if et not in VALID_ENCODER_TYPES:
        raise ValueError(
            f"unknown encoder_type {encoder_type!r}; "
            f"expected one of {sorted(VALID_ENCODER_TYPES)}"
        )
    if et == "risk":
        fw_mu = [list(fw.get_daily_demand_means()) for fw in forward_warehouses]
        return RiskFeatureEncoder(
            fw_daily_demand_means=fw_mu,
            evaluation_days=evaluation_days,
            num_products=len(central_warehouse.order_up_to_levels),
        )
    forward_s = [list(fw.order_up_to_levels) for fw in forward_warehouses]
    return StateEncoder(
        central_order_up_to=list(central_warehouse.order_up_to_levels),
        forward_order_up_to=forward_s,
        evaluation_days=evaluation_days,
        num_products=len(central_warehouse.order_up_to_levels),
    )


def build_state_encoder_from_env(
    env,
    *,
    encoder_type: str = "raw",
) -> EncoderLike:
    """Convenience: call after env.reset() so engine warehouses exist."""
    if env.engine is None:
        raise RuntimeError("call env.reset() before build_state_encoder_from_env")
    return build_state_encoder_from_warehouses(
        env.engine.central_warehouse,
        env.engine.forward_warehouses,
        env.config.evaluation_days,
        encoder_type=encoder_type,
    )
