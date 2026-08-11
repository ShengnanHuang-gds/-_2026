from __future__ import annotations

import math
from typing import Dict, List

from config.simulation_config import SimulationConfig
from models.forward_warehouse import ForwardWarehouse
from models.product import Product


PRODUCT_LEVEL_METRICS = [
    "central_shortage_frequency_by_product",
    "central_zero_inventory_frequency_by_product",
    "forward_stockout_frequency_by_product",
]


class PerformanceTracker:
    """step7 Collect daily snapshots and compute KPIs ."""

    def __init__(
        self,
        config: SimulationConfig,
        products: List[Product],
    ) -> None:
        self.config = config
        self.products = products
        self.num_products = config.num_products
        self.num_forward_warehouses = config.num_forward_warehouses

        # 累计量（只累 evaluation 天）
        self.total_demand = 0
        self.total_sales = 0
        self.total_lost_sales = 0
        self.total_profit = 0.0
        self.total_expedite_cost = 0.0

        self.total_average_forward_inventory = 0.0
        self.total_average_central_inventory = 0.0
        self.evaluation_day_count = 0

        self.forward_capacity_utilizations: List[float] = []
        self.central_capacity_utilizations: List[float] = []

        # 每个产品 CW shortage 次数
        self.central_shortage_counts = [0] * self.num_products
        # 每个产品 CW 日末库存为 0 的天数
        self.central_zero_inventory_counts = [0] * self.num_products
        # 每个产品 FW 缺货天数（任一 FW 当日 lost_sales > 0 则计 1 次）
        self.forward_stockout_counts = [0] * self.num_products

    def log_daily_snapshot(
        self,
        snapshot: dict,
        forward_warehouses: List[ForwardWarehouse],
    ) -> None:
        """Step 7: record one day's performance."""
        if snapshot["is_warmup"]:
            return


        central_inventory_begin = snapshot["central_inventory_begin"]  # IC^b
        central_inventory_end = snapshot["central_inventory_end"]      # IC^e
        forward_inventory_begin = snapshot["forward_inventory_begin"]  # IF^b
        forward_inventory_end = snapshot["forward_inventory_end"]      # IF^e
        shortage_flags = snapshot["shortage_flags"]

        daily_revenue = 0.0
        daily_penalty = 0.0
        daily_demand = 0
        daily_sales = 0
        daily_lost_sales = 0

        for forward_warehouse in forward_warehouses:
            forward_warehouse_id = forward_warehouse.forward_warehouse_id

            for product_index, product in enumerate(self.products):
                demand = forward_warehouse.today_demand[product_index]
                sales = forward_warehouse.today_sales[product_index]
                lost_sales = forward_warehouse.today_lost_sales[product_index]

                daily_demand += demand
                daily_sales += sales
                daily_lost_sales += lost_sales

                #section12 daily revenue
                daily_revenue += product.selling_price * sales
                #section12 daily lost-sales penalty
                daily_penalty += product.lost_sales_penalty * lost_sales

                average_forward_inventory = (
                    forward_inventory_begin[forward_warehouse_id][product_index]
                    + forward_inventory_end[forward_warehouse_id][product_index]
                ) / 2
                self.total_average_forward_inventory += average_forward_inventory

        daily_holding_cost = 0.0

        for product_index, product in enumerate(self.products):
            #cw daily holding cost
            average_central_inventory = (
                central_inventory_begin[product_index]
                + central_inventory_end[product_index]
            ) / 2
            self.total_average_central_inventory += average_central_inventory

            daily_holding_cost += product.central_holding_cost * average_central_inventory

            for forward_warehouse in forward_warehouses:
                #fw daily holding cost
                forward_warehouse_id = forward_warehouse.forward_warehouse_id
                average_forward_inventory = (
                    forward_inventory_begin[forward_warehouse_id][product_index]
                    + forward_inventory_end[forward_warehouse_id][product_index]
                ) / 2
                # daily holding cost
                daily_holding_cost += (
                    product.forward_holding_cost * average_forward_inventory
                )

        # Daily profit (PDF reward subtracts expedite / emergency action cost)
        expedite_cost = float(snapshot.get("expedite_cost", 0.0) or 0.0)
        daily_profit = (
            daily_revenue - daily_holding_cost - daily_penalty - expedite_cost
        )

        self.total_demand += daily_demand
        self.total_sales += daily_sales
        self.total_lost_sales += daily_lost_sales
        self.total_profit += daily_profit
        self.total_expedite_cost += expedite_cost
        self.evaluation_day_count += 1

       #Capacity Utilization 13.4
        for forward_warehouse in forward_warehouses:
            forward_warehouse_id = forward_warehouse.forward_warehouse_id
            total_forward_inventory = sum(forward_inventory_begin[forward_warehouse_id])
            utilization = total_forward_inventory / self.config.forward_warehouse_capacity
            self.forward_capacity_utilizations.append(utilization)

        total_central_inventory = sum(central_inventory_begin)
        central_utilization = (
            total_central_inventory / self.config.central_warehouse_capacity
        )
        self.central_capacity_utilizations.append(central_utilization)

        # --- CW shortage frequency ---
        for product_index, shortage in enumerate(shortage_flags):
            if shortage:
                self.central_shortage_counts[product_index] += 1

        # --- CW zero-inventory days (end-of-day on-hand) ---
        for product_index, inventory_level in enumerate(central_inventory_end):
            if inventory_level == 0:
                self.central_zero_inventory_counts[product_index] += 1

        # --- FW stockout days (lost sales during demand fulfillment) ---
        for forward_warehouse in forward_warehouses:
            for product_index in range(self.num_products):
                if forward_warehouse.today_lost_sales[product_index] > 0:
                    self.forward_stockout_counts[product_index] += 1

    def summarize(self) -> dict:
        """Return replication-level metrics."""
        evaluation_days = self.evaluation_day_count
        if evaluation_days == 0:
            raise ValueError("No evaluation days recorded")

        num_forward_warehouses = self.num_forward_warehouses
        num_products = self.num_products

        fill_rate = self.total_sales / self.total_demand if self.total_demand > 0 else 0.0
        lost_sales_rate = (
            self.total_lost_sales / self.total_demand if self.total_demand > 0 else 0.0
        )

        average_forward_inventory = (
            self.total_average_forward_inventory
            / (num_forward_warehouses * num_products * evaluation_days)
        )
        average_central_inventory = (
            self.total_average_central_inventory / (num_products * evaluation_days)
        )

        return {
            "evaluation_days": evaluation_days,
            "total_profit": self.total_profit,
            "total_expedite_cost": self.total_expedite_cost,
            "fill_rate": fill_rate,
            "lost_sales_rate": lost_sales_rate,
            "average_forward_inventory": average_forward_inventory,
            "average_central_inventory": average_central_inventory,
            "forward_capacity_utilization_mean": _mean(
                self.forward_capacity_utilizations
            ),
            "forward_capacity_utilization_max": _max(
                self.forward_capacity_utilizations
            ),
            "forward_capacity_utilization_p95": _percentile(
                self.forward_capacity_utilizations, 95
            ),
            "central_capacity_utilization_mean": _mean(
                self.central_capacity_utilizations
            ),
            "central_capacity_utilization_max": _max(
                self.central_capacity_utilizations
            ),
            "central_capacity_utilization_p95": _percentile(
                self.central_capacity_utilizations, 95
            ),
            "central_shortage_frequency_by_product": [
                count / evaluation_days
                for count in self.central_shortage_counts
            ],
            "central_zero_inventory_frequency_by_product": [
                count / evaluation_days
                for count in self.central_zero_inventory_counts
            ],
            "forward_stockout_frequency_by_product": [
                count / (num_forward_warehouses * evaluation_days)
                for count in self.forward_stockout_counts
            ],
        }

    @staticmethod
    def report_final_metrics(replication_results: List[dict]) -> dict:
        """Compute mean and 95% CI across replications."""
        num_replications = len(replication_results)
        if num_replications == 0:
            raise ValueError("No replication results provided")

        metric_names = [
            key
            for key in replication_results[0].keys()
            if key not in PRODUCT_LEVEL_METRICS
        ]

        summary = {}

        for metric_name in metric_names:
            values = [result[metric_name] for result in replication_results]
            summary[metric_name] = _mean_and_ci(values, num_replications)

        for metric_name in PRODUCT_LEVEL_METRICS:
            if metric_name not in replication_results[0]:
                continue
            num_products = len(replication_results[0][metric_name])
            product_summary = []

            for product_index in range(num_products):
                values = [
                    result[metric_name][product_index]
                    for result in replication_results
                ]
                product_summary.append(_mean_and_ci(values, num_replications))

            summary[metric_name] = product_summary

        return summary


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _max(values: List[float]) -> float:
    return max(values) if values else 0.0


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = math.ceil(percentile / 100 * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def _mean_and_ci(values: List[float], num_replications: int) -> dict:
    mean_value = _mean(values)
    if num_replications <= 1:
        return {"mean": mean_value, "ci_half_width": 0.0}

    variance = sum((value - mean_value) ** 2 for value in values) / (num_replications - 1)
    std_dev = math.sqrt(variance)
    ci_half_width = 1.96 * std_dev / math.sqrt(num_replications)

    return {
        "mean": mean_value,
        "ci_half_width": ci_half_width,
    }
