from __future__ import annotations

import math

from models.product import Product


def compute_daily_demand_mean(
    demand_intensity_multiplier: float,
    product: Product,
) -> float:
    """section5 demand model"""
    return demand_intensity_multiplier * product.base_daily_demand


def compute_forward_warehouse_reorder_point(daily_demand_mean: float) -> int:
    """section10 s"""
    return math.ceil(
        daily_demand_mean + 1.28 * math.sqrt(daily_demand_mean)
    )


def compute_forward_warehouse_order_up_to_level(daily_demand_mean: float) -> int:
    """section 10 S"""
    return math.ceil(
        3 * daily_demand_mean + 1.64 * math.sqrt(3 * daily_demand_mean)
    )


def compute_forward_warehouse_policy(
    demand_intensity_multiplier: float,
    products: list[Product],
) -> tuple[list[int], list[int]]:
    """
    Return a tuple contain two list 
    first list is the s for 5 product
    another list is S for 5 product
    """
    reorder_points = []
    order_up_to_levels = []

    for product in products:
        daily_demand_mean = compute_daily_demand_mean(
            demand_intensity_multiplier, product
        )
        reorder_points.append(
            compute_forward_warehouse_reorder_point(daily_demand_mean)
        )
        order_up_to_levels.append(
            compute_forward_warehouse_order_up_to_level(daily_demand_mean)
        )

    return reorder_points, order_up_to_levels


def compute_network_daily_demand_mean(product: Product) -> float:
    """
    section 11 default aggregate daily demand
    """
    return 11 * product.base_daily_demand


def compute_central_warehouse_reorder_point(network_daily_demand_mean: float) -> int:
    """section11 CW s"""
    return math.ceil(
        3 * network_daily_demand_mean
        + 1.64 * math.sqrt(3 * network_daily_demand_mean)
    )


def compute_central_warehouse_order_up_to_level(
    network_daily_demand_mean: float,
) -> int:
    """section 12 cw S"""
    return math.ceil(
        7 * network_daily_demand_mean
        + 1.64 * math.sqrt(7 * network_daily_demand_mean)
    )


def compute_central_warehouse_policy(
    products: list[Product],
) -> tuple[list[int], list[int]]:
    """Return a tuple containing 2 list each list consist of s S for each product for the CW."""
    reorder_points = []
    order_up_to_levels = []

    for product in products:
        network_daily_demand_mean = compute_network_daily_demand_mean(product)
        reorder_points.append(
            compute_central_warehouse_reorder_point(network_daily_demand_mean)
        )
        order_up_to_levels.append(
            compute_central_warehouse_order_up_to_level(network_daily_demand_mean)
        )

    return reorder_points, order_up_to_levels

