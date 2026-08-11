from __future__ import annotations

import random

import pytest

from config.disruption_chain import (
    EXTRA_DELAY,
    M_A,
    M_U,
    RELEASE_RATIO,
    DisruptionState,
    SupplierAvailability,
    TransportState,
    compute_realized_lead_time,
    default_disruption_state,
    get_transition_matrices,
    mean_sojourn_time,
    released_quantity,
    sample_next_disruption_state,
    sample_next_supplier_availability,
    sample_next_transport_state,
    validate_all_matrix_profiles,
    validate_transition_matrices,
)


def test_release_ratios_match_pdf():
    assert RELEASE_RATIO[SupplierAvailability.A0] == 1.00
    assert RELEASE_RATIO[SupplierAvailability.A1] == 0.80
    assert RELEASE_RATIO[SupplierAvailability.A2] == 0.40
    assert RELEASE_RATIO[SupplierAvailability.A3] == 0.00


def test_extra_delays_match_pdf():
    assert EXTRA_DELAY[TransportState.U0] == 0
    assert EXTRA_DELAY[TransportState.U1] == 1
    assert EXTRA_DELAY[TransportState.U2] == 2


def test_transition_matrices_are_row_stochastic():
    validate_transition_matrices()


def test_all_matrix_profiles_are_row_stochastic():
    validate_all_matrix_profiles()


def test_get_transition_matrices_pdf_matches_module_defaults():
    m_a, m_u = get_transition_matrices("pdf")
    assert m_a == M_A
    assert m_u == M_U


def test_harsh_profiles_leave_normal_faster_than_pdf():
    m_a_pdf, m_u_pdf = get_transition_matrices("pdf")
    m_a_harsh, m_u_harsh = get_transition_matrices("harsh_mild")
    assert m_a_harsh[0][0] < m_a_pdf[0][0]
    assert m_u_harsh[0][0] < m_u_pdf[0][0]


def test_unknown_matrix_profile_raises():
    with pytest.raises(ValueError, match="unknown disruption_matrix_profile"):
        get_transition_matrices("not_a_real_profile")


def test_mean_sojourn_times_match_phase2_notes():
    assert abs(mean_sojourn_time(M_A[0][0]) - 200.0) < 1e-9
    assert abs(mean_sojourn_time(M_A[1][1]) - 2.5) < 1e-9
    assert abs(mean_sojourn_time(M_A[2][2]) - (1.0 / 0.45)) < 1e-9
    assert abs(mean_sojourn_time(M_A[3][3]) - (1.0 / 0.35)) < 1e-9

    assert abs(mean_sojourn_time(M_U[0][0]) - 125.0) < 1e-9
    assert abs(mean_sojourn_time(M_U[1][1]) - 2.0) < 1e-9
    assert abs(mean_sojourn_time(M_U[2][2]) - (1.0 / 0.55)) < 1e-9
    assert abs(mean_sojourn_time(M_U[3][3]) - 2.0) < 1e-9


def test_default_state_is_normal():
    state = default_disruption_state()
    assert state.supplier_availability == SupplierAvailability.A0
    assert state.transport_state == TransportState.U0
    assert state.code() == "Z_00"
    assert state.release_ratio == 1.0
    assert state.extra_delay == 0
    assert not state.is_transport_blocked


def test_released_quantity_examples():
    assert released_quantity(100, SupplierAvailability.A0) == 100
    assert released_quantity(100, SupplierAvailability.A1) == 80
    assert released_quantity(100, SupplierAvailability.A2) == 40
    assert released_quantity(100, SupplierAvailability.A3) == 0


def test_realized_lead_time_examples():
    assert compute_realized_lead_time(2, TransportState.U0) == 2
    assert compute_realized_lead_time(2, TransportState.U1) == 3
    assert compute_realized_lead_time(2, TransportState.U2) == 4
    assert compute_realized_lead_time(2, TransportState.U1, expedite=1) == 2
    assert compute_realized_lead_time(1, TransportState.U0, expedite=1) == 1


def test_u3_has_no_finite_lead_time():
    state = DisruptionState(
        supplier_availability=SupplierAvailability.A0,
        transport_state=TransportState.U3,
    )
    assert state.is_transport_blocked
    try:
        _ = state.extra_delay
        assert False, "expected ValueError for U3 extra_delay"
    except ValueError:
        pass

    try:
        compute_realized_lead_time(2, TransportState.U3)
        assert False, "expected ValueError for U3 lead time"
    except ValueError:
        pass


def test_sampling_returns_valid_states():
    random_generator = random.Random(42)
    availability = SupplierAvailability.A0
    transport = TransportState.U0

    for _ in range(500):
        availability = sample_next_supplier_availability(
            availability, random_generator
        )
        transport = sample_next_transport_state(transport, random_generator)
        assert availability in SupplierAvailability
        assert transport in TransportState


def test_joint_sampling_stays_near_normal_from_z00():
    """From Z_00, most next states should remain A0 and U0."""
    random_generator = random.Random(0)
    state = default_disruption_state()
    same_a0 = 0
    same_u0 = 0
    trials = 2000

    for _ in range(trials):
        nxt = sample_next_disruption_state(state, random_generator)
        if nxt.supplier_availability == SupplierAvailability.A0:
            same_a0 += 1
        if nxt.transport_state == TransportState.U0:
            same_u0 += 1

    assert same_a0 / trials > 0.98
    assert same_u0 / trials > 0.98


def test_harsh_mild_sampling_leaves_normal_more_often():
    m_a, m_u = get_transition_matrices("harsh_mild")
    rng = random.Random(0)
    state = default_disruption_state()
    leave_a0 = 0
    trials = 5000
    for _ in range(trials):
        nxt = sample_next_disruption_state(state, rng, m_a=m_a, m_u=m_u)
        if nxt.supplier_availability != SupplierAvailability.A0:
            leave_a0 += 1
    # PDF leave rate ≈ 0.5%; harsh_mild ≈ 3%
    assert 0.015 < leave_a0 / trials < 0.05
