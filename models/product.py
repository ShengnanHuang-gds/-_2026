class Product:
    """Static economic parameters for one product (PDF Table 2)."""

    def __init__(
        self,
        product_id: int,
        product_name: str,
        base_daily_demand: float,
        selling_price: float,
        central_holding_cost: float,
        forward_holding_cost: float,
        lost_sales_penalty: float,
    ) -> None:
        self.product_id = product_id
        self.product_name = product_name
        self.base_daily_demand = base_daily_demand  # lambda_bar_k
        self.selling_price = selling_price  # p_k
        self.central_holding_cost = central_holding_cost  # h^C_k
        self.forward_holding_cost = forward_holding_cost  # h^F_k
        self.lost_sales_penalty = lost_sales_penalty  # b_k

    def __repr__(self) -> str:
        return (
            f"Product(product_id={self.product_id}, product_name={self.product_name!r}, "
            f"base_daily_demand={self.base_daily_demand}, "
            f"selling_price={self.selling_price})"
        )
