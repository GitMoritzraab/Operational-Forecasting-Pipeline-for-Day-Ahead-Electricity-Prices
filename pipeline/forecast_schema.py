"""Dependency-free constants for persisted forecast result contracts."""

FORECAST_SCHEMA_VERSION = 2
FORECAST_INDEX_NAME = "delivery_date_mtu"
DELIVERY_DATE_COLUMN = "delivery_date"
MTU_COLUMN = "mtu"
INDEX_COLUMNS = (DELIVERY_DATE_COLUMN, MTU_COLUMN)
MTUS_PER_DAY = 96
