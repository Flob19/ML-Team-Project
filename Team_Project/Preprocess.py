import pandas as pd
from pathlib import Path

DATA = Path("Data")
sales_fp = DATA / "Coffee Shop Sales.xlsx"
weather_fp = DATA / "NY_Weather_data_2023_jan-juni.xlsx"
out_tx_fp = DATA / "sales_with_weather_tx.xlsx"
out_hourly_fp = DATA / "hourly_orders_with_weather.xlsx"


df_sales = pd.read_excel(sales_fp)


dt = pd.to_datetime(df_sales["transaction_date"].astype(str) + " " + df_sales["transaction_time"].astype(str))
df_sales["datetime"] = dt
df_sales["hour"] = df_sales["datetime"].dt.floor("h")

df_weather = pd.read_excel(weather_fp, skiprows=3)


df_weather = df_weather.rename(columns={
    "time": "hour",
    "temperature_2m (°C)": "temperature_C",          
    "rain (mm)": "rain_mm",
    "snowfall (cm)": "snow_cm",
    "cloud_cover (%)": "cloud_cover_pct",
    "wind_speed_10m (km/h)": "wind_speed_kmh"
})

df_weather["hour"] = pd.to_datetime(df_weather["hour"])

df_merged_tx = pd.merge(
    df_sales,
    df_weather[["hour", "temperature_C", "rain_mm", "snow_cm", "cloud_cover_pct", "wind_speed_kmh"]],
    on="hour",
    how="left"
)

df_merged_tx.to_excel(out_tx_fp, index=False)


df_hourly = (
    df_merged_tx
      .groupby("hour", as_index=False)
      .agg(orders=("transaction_id", "count"),
           temperature_C=("temperature_C", "mean"),
           rain_mm=("rain_mm", "mean"),
           snow_cm=("snow_cm", "mean"),
           cloud_cover_pct=("cloud_cover_pct", "mean"),
           wind_speed_kmh=("wind_speed_kmh", "mean"))
)

df_hourly.to_excel(out_hourly_fp, index=False)
print("Klart! Skapade:")
print(" -", out_tx_fp)
print(" -", out_hourly_fp)
