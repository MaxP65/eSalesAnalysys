# EDA Report: initial_eda

Rows: 96476
Columns: 21

## Dataset Overview

- Time column: `order_purchase_timestamp`
- Target column: `is_long_delivery`
- Numeric columns: 13
- Categorical columns: 7
- Duplicate ratio: 0.0000

## Missing Values

- `product_category_name_english`: 0.0436
- `payment_type_main`: 0.0300
- `freight_value_sum`: 0.0300
- `payments_count`: 0.0000
- `payment_installments_max`: 0.0000
- `payment_value_sum`: 0.0000

## Target Distribution

- `0`: 0.7500
- `1`: 0.2500

## Figures

- ![](figures/initial_eda_orders_over_time.png)
- ![](figures/initial_eda_target_distribution.png)
- ![](figures/initial_eda_long_delivery_by_month.png)
- ![](figures/initial_eda_customer_state_long_delivery_ratio.png)
- ![](figures/initial_eda_seller_state_long_delivery_ratio.png)
- ![](figures/initial_eda_product_category_name_english_long_delivery_ratio.png)
- ![](figures/initial_eda_long_delivery_by_weekday.png)
- ![](figures/initial_eda_long_delivery_by_same_state.png)
- ![](figures/initial_eda_price_sum_distribution.png)
- ![](figures/initial_eda_freight_value_sum_distribution.png)
- ![](figures/initial_eda_items_count_distribution.png)
- ![](figures/initial_eda_delivery_time_days_distribution.png)
- ![](figures/initial_eda_price_sum_by_target.png)
- ![](figures/initial_eda_freight_value_sum_by_target.png)
- ![](figures/initial_eda_correlation_heatmap.png)
