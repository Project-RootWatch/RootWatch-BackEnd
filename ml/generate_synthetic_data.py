"""
Generates a synthetic soil-moisture time series to learn/prototype the LSTM
pipeline on, since our real sensor data isn't usable yet (see the
conversation this came from: 49 real rows, 45 of them stuck at 100% due to
an uncalibrated sensor, plus a 5.5-day gap with no data at all).

The physics we're faking, in plain terms:
  - Soil moisture decays between waterings (evaporation + the plant
    drinking it). Decay speeds up when it's hotter and brighter.
  - When moisture drops below a threshold, "irrigation" fires and moisture
    jumps back up — this produces the repeating sawtooth pattern real
    irrigated soil actually shows, which is exactly the pattern we want
    the model to learn to anticipate.
  - Temperature and light both follow a daily cycle (cool nights, warm
    bright middays), with slow day-to-day "weather" drift layered on top.

Swap this file for a real data export once the sensor is fixed and enough
history has accumulated — nothing downstream needs to change, since it
only cares about the CSV's three columns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta, timezone

OUT_DIR = Path(__file__).parent / "data"
STEPS_PER_DAY = 96  # 24h / 15min
N_DAYS = 45
N = N_DAYS * STEPS_PER_DAY
SEED = 42


def generate():
    rng = np.random.default_rng(SEED)

    t = np.arange(N)
    hour = (t * 15 / 60) % 24  # fractional hour-of-day, 0-24
    day = t // STEPS_PER_DAY

    # ---- Temperature: daily cycle + slow multi-day weather drift + noise ----
    daily_weather = np.cumsum(rng.normal(0, 1.0, N_DAYS))
    daily_weather -= daily_weather.mean()
    weather_per_step = np.repeat(daily_weather, STEPS_PER_DAY)

    temperature = (
        26                                      # mean temp
        + 6 * np.cos(2 * np.pi * (hour - 14) / 24)  # daily cycle, peak at 2pm
        + weather_per_step * 0.5                # slow drift across days
        + rng.normal(0, 0.3, N)                 # sensor noise
    )

    # ---- Light: 0 at night, bell curve peaking at noon, cloudy-day damping ----
    daylight = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    cloud_factor = np.repeat(rng.uniform(0.6, 1.0, N_DAYS), STEPS_PER_DAY)
    light = daylight * 100 * cloud_factor + rng.normal(0, 2, N)
    light = np.clip(light, 0, 100)

    # ---- Soil moisture: decay + irrigation resets (sequential simulation) ----
    moisture = np.zeros(N)
    moisture[0] = 65.0
    irrigation_threshold = 30.0

    for i in range(1, N):
        base_decay = 0.08
        temp_effect = max(0.0, temperature[i] - 20) * 0.015
        light_effect = (light[i] / 100) * 0.05
        decay = base_decay + temp_effect + light_effect

        value = moisture[i - 1] - decay + rng.normal(0, 0.15)

        if value < irrigation_threshold:
            value = rng.uniform(85, 95)  # irrigation just fired

        moisture[i] = np.clip(value, 0, 100)

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(minutes=15 * i) for i in range(N)]

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "soil_moisture": moisture.round(2),
            "temperature": temperature.round(2),
            "light_level": light.round(2),
        }
    )
    return df


def plot_preview(df, days=5):
    subset = df.iloc[: days * STEPS_PER_DAY]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(subset["timestamp"], subset["soil_moisture"], color="#f2a93c")
    axes[0].set_ylabel("Soil moisture (%)")
    axes[0].axhline(30, color="#888", linestyle="--", linewidth=0.8, label="irrigation threshold")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(subset["timestamp"], subset["temperature"], color="#e34948")
    axes[1].set_ylabel("Temp (°C)")

    axes[2].plot(subset["timestamp"], subset["light_level"], color="#2a78d6")
    axes[2].set_ylabel("Light (%)")
    axes[2].set_xlabel("Time")

    fig.suptitle(f"Synthetic RootWatch data — first {days} days")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "synthetic_preview.png", dpi=130)
    print(f"Saved preview plot to {OUT_DIR / 'synthetic_preview.png'}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    csv_path = OUT_DIR / "synthetic_readings.csv"
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} rows spanning {N_DAYS} days -> {csv_path}")
    print(df.describe())
    plot_preview(df)
