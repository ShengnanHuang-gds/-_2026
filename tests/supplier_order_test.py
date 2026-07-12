from __future__ import annotations

import pytest

from models.supplier_order import SupplierOrder


def test_supplier_order_stores_valid_values():
    order = SupplierOrder(
        quantities=[100, 50, 0, 20, 0],
        remaining_lead_time=2,
    )

    assert order.quantities == [100, 50, 0, 20, 0]
    assert order.remaining_lead_time == 2


def test_supplier_order_allows_all_zero_quantities():
    order = SupplierOrder(
        quantities=[0, 0, 0, 0, 0],
        remaining_lead_time=1,
    )

    assert order.quantities == [0, 0, 0, 0, 0]
    assert order.remaining_lead_time == 1


def test_supplier_order_copies_quantities_list():
    original_quantities = [10, 0, 0, 0, 0]
    order = SupplierOrder(
        quantities=original_quantities,
        remaining_lead_time=3,
    )

    original_quantities[0] = 999

    assert order.quantities == [10, 0, 0, 0, 0]


def test_supplier_order_rejects_wrong_number_of_products():
    with pytest.raises(ValueError, match="exactly 5"):
        SupplierOrder(quantities=[1, 2, 3], remaining_lead_time=2)


def test_supplier_order_rejects_zero_lead_time():
    with pytest.raises(ValueError, match=">= 1"):
        SupplierOrder(quantities=[1, 0, 0, 0, 0], remaining_lead_time=0)


def test_supplier_order_rejects_negative_lead_time():
    with pytest.raises(ValueError, match=">= 1"):
        SupplierOrder(quantities=[1, 0, 0, 0, 0], remaining_lead_time=-1)


def test_supplier_order_rejects_negative_quantities():
    with pytest.raises(ValueError, match="non-negative"):
        SupplierOrder(quantities=[1, -2, 0, 0, 0], remaining_lead_time=2)


def test_supplier_order_repr():
    order = SupplierOrder(
        quantities=[10, 0, 5, 0, 0],
        remaining_lead_time=2,
    )

    assert repr(order) == (
        "SupplierOrder(quantities=[10, 0, 5, 0, 0], "
        "remaining_lead_time=2)"
    )