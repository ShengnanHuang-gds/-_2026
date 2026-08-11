from __future__ import annotations

from config.disruption_chain import (
    DisruptionState,
    SupplierAvailability,
    TransportState,
)
from models.supplier_upstream import SupplierUpstreamBuffers
from tests.test_central_warehouse import make_central_warehouse


def _order_100_p1() -> list[int]:
    return [100, 0, 0, 0, 0]


def test_z00_releases_all_into_pipeline_with_lead_time_2():
    upstream = SupplierUpstreamBuffers()
    state = DisruptionState(
        supplier_availability=SupplierAvailability.A0,
        transport_state=TransportState.U0,
    )

    order = upstream.process_evening(_order_100_p1(), state, base_lead_time=2)

    assert upstream.supplier_backlog == [0, 0, 0, 0, 0]
    assert upstream.transport_waiting == [0, 0, 0, 0, 0]
    assert order is not None
    assert order.quantities == [100, 0, 0, 0, 0]
    assert order.remaining_lead_time == 2


def test_z21_releases_40_with_lead_time_3_and_keeps_60_backlog():
    upstream = SupplierUpstreamBuffers()
    state = DisruptionState(
        supplier_availability=SupplierAvailability.A2,
        transport_state=TransportState.U1,
    )

    order = upstream.process_evening(_order_100_p1(), state, base_lead_time=2)

    assert upstream.supplier_backlog == [60, 0, 0, 0, 0]
    assert upstream.transport_waiting == [0, 0, 0, 0, 0]
    assert order is not None
    assert order.quantities == [40, 0, 0, 0, 0]
    assert order.remaining_lead_time == 3


def test_z33_releases_nothing_and_leaves_all_in_backlog():
    upstream = SupplierUpstreamBuffers()
    state = DisruptionState(
        supplier_availability=SupplierAvailability.A3,
        transport_state=TransportState.U3,
    )

    order = upstream.process_evening(_order_100_p1(), state, base_lead_time=2)

    assert order is None
    assert upstream.supplier_backlog == [100, 0, 0, 0, 0]
    assert upstream.transport_waiting == [0, 0, 0, 0, 0]


def test_z13_releases_80_into_waiting_under_transport_block():
    upstream = SupplierUpstreamBuffers()
    state = DisruptionState(
        supplier_availability=SupplierAvailability.A1,
        transport_state=TransportState.U3,
    )

    order = upstream.process_evening(_order_100_p1(), state, base_lead_time=2)

    assert order is None
    assert upstream.supplier_backlog == [20, 0, 0, 0, 0]
    assert upstream.transport_waiting == [80, 0, 0, 0, 0]


def test_waiting_dispatches_when_transport_recovers():
    upstream = SupplierUpstreamBuffers()
    blocked = DisruptionState(
        supplier_availability=SupplierAvailability.A1,
        transport_state=TransportState.U3,
    )
    recovered = DisruptionState(
        supplier_availability=SupplierAvailability.A0,
        transport_state=TransportState.U0,
    )

    assert upstream.process_evening(_order_100_p1(), blocked, base_lead_time=2) is None
    assert upstream.transport_waiting == [80, 0, 0, 0, 0]
    assert upstream.supplier_backlog == [20, 0, 0, 0, 0]

    # No new order: release remaining backlog under A0, then dispatch all waiting.
    order = upstream.process_evening([0, 0, 0, 0, 0], recovered, base_lead_time=2)

    assert upstream.supplier_backlog == [0, 0, 0, 0, 0]
    assert upstream.transport_waiting == [0, 0, 0, 0, 0]
    assert order is not None
    assert order.quantities == [100, 0, 0, 0, 0]
    assert order.remaining_lead_time == 2


def test_central_warehouse_z21_ordering_path():
    cw = make_central_warehouse()
    cw.inventory = [0, 0, 0, 0, 0]
    cw.set_disruption_state(
        DisruptionState(
            supplier_availability=SupplierAvailability.A2,
            transport_state=TransportState.U1,
        )
    )

    # Force a single-SKU order of 100 through upstream by placing it directly,
    # then run evening processing with fixed L0 = 2 via a stub RNG path:
    # use process_evening on cw.upstream to keep the assertion focused.
    order = cw.upstream.process_evening(
        _order_100_p1(),
        cw.disruption_state,
        base_lead_time=2,
    )
    if order is not None:
        cw.supplier_pipeline.append(order)

    assert cw.upstream.supplier_backlog[0] == 60
    assert len(cw.supplier_pipeline) == 1
    assert cw.supplier_pipeline[0].quantities[0] == 40
    assert cw.supplier_pipeline[0].remaining_lead_time == 3


def test_inventory_position_includes_backlog_and_waiting():
    cw = make_central_warehouse()
    cw.inventory[0] = 10
    cw.upstream.supplier_backlog[0] = 20
    cw.upstream.transport_waiting[0] = 5
    cw.supplier_pipeline = []

    assert cw.get_inventory_position(0) == 35
