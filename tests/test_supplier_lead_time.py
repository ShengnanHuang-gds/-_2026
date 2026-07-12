from __future__ import annotations

import random

from config.simulation_config import SimulationConfig
from config.supplier_lead_time import SupplierLeadTimeMode


def test_baseline_lead_time_in_1_to_3():
    config = SimulationConfig(supplier_lead_time_mode=SupplierLeadTimeMode.BASELINE)
    random_generator = random.Random(0)

    for _ in range(200):
        lead_time = config.sample_supplier_lead_time(random_generator)
        assert 1 <= lead_time <= 3


def test_volatile_lead_time_in_1_to_5():
    config = SimulationConfig(supplier_lead_time_mode=SupplierLeadTimeMode.VOLATILE)
    random_generator = random.Random(1)

    for _ in range(200):
        lead_time = config.sample_supplier_lead_time(random_generator)
        assert 1 <= lead_time <= 5


def test_disruption_mixture_includes_long_lead_times():
    config = SimulationConfig(
        supplier_lead_time_mode=SupplierLeadTimeMode.DISRUPTION_MIXTURE
    )
    random_generator = random.Random(42)

    samples = [
        config.sample_supplier_lead_time(random_generator) for _ in range(10000)
    ]

    assert any(lead_time >= 4 for lead_time in samples)
    assert all(1 <= lead_time <= 6 for lead_time in samples)
