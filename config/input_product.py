from __future__ import annotations

from models.product import Product

# PDF Table 3: forward warehouse demand intensity multipliers (rho_i)
DEFAULT_FORWARD_WAREHOUSE_DEMAND_INTENSITY_MULTIPLIERS = [
    0.75, 0.85, 0.95, 1.00, 1.05,
    1.10, 1.20, 1.25, 1.35, 1.50,
]

DEFAULT_PRODUCT_ROWS = [
    (1, "P1", 12, 10, 0.10, 5.00),
    (2, "P2", 8, 15, 0.15, 7.50),
    (3, "P3", 5, 25, 0.25, 12.50),
    (4, "P4", 3, 40, 0.40, 20.00),
    (5, "P5", 2, 60, 0.60, 30.00),
]


def get_default_products(forward_holding_cost_ratio: float = 3.0) -> list[Product]:
    """Return 5 default products from PDF Table 2."""
    return [
        Product(
            product_id=product_id,
            product_name=product_name,
            base_daily_demand=base_daily_demand,
            selling_price=selling_price,
            central_holding_cost=central_holding_cost,
            forward_holding_cost=central_holding_cost * forward_holding_cost_ratio,
            lost_sales_penalty=lost_sales_penalty,
        )
        for product_id, product_name, base_daily_demand, selling_price,
            central_holding_cost, lost_sales_penalty in DEFAULT_PRODUCT_ROWS
    ]


def get_default_forward_warehouse_demand_intensity_multipliers() -> list[float]:
    """Return demand intensity multipliers for FW 1..10 (PDF Table 3, rho_i)."""
    return list(DEFAULT_FORWARD_WAREHOUSE_DEMAND_INTENSITY_MULTIPLIERS)
