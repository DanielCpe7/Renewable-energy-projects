from windpowerlib import wind_farm, WindTurbine, ModelChain

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# === 1. Definir turbinas de la librería ===
turbinas_config = {
    "E-126/4200": {"hub_height": 135, "rotor_diameter": 126, "turbine_type": "E-126/4200"},
    "E48/800":    {"hub_height": 50,  "rotor_diameter": 48,  "turbine_type": "E48/800"},
    "V126/3000":  {"hub_height": 117, "rotor_diameter": 126, "turbine_type": "V126/3000"},
}


# === 1b. Definir turbina personalizada ===
data = {
    "wind_speed": [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 20, 25],
    "value":      [0, 0, 200e3, 400e3, 500e3, 800e3, 800e3, 800e3,
                   800e3, 800e3, 800e3, 800e3, 800e3, 800e3, 800e3]
}
power_curve = pd.DataFrame(data)

mi_turbina = WindTurbine(
    name="MiTurbina/800",
    hub_height=50,
    rotor_diameter=48,
    nominal_power=800e3,  # 800 kW
    power_curve=power_curve
)


# === 2. Leer datos meteorológicos ===
try:
    weather = pd.read_csv(
        "Riohacha1.csv",
        sep=";",
        index_col=0,
        parse_dates=True,
        dayfirst=True,
        encoding="latin1"
    )

    if not isinstance(weather.index, pd.DatetimeIndex):
        weather.index = pd.to_datetime(weather.index, dayfirst=True)

except Exception as e:
    print(f"Error leyendo el archivo: {e}")

    # Datos de ejemplo
    data = {
        'wind_speed_ms': [1.2, 0.9, 1.3, 0.7, 1.2, 1.5, 1.3, 1.6, 1.3, 0.9],
        'temperature_C': [29.0, 29.6, 32.2, 31.6, 31.6, 31.8, 29.4, 33.2, 33.1, 34.7],
        'pressure_Pa': [101300] * 10
    }

    dates = pd.date_range(start='2024-08-01 00:00', periods=10, freq='1H')
    weather = pd.DataFrame(data, index=dates)

    print("Usando datos de ejemplo")


# Renombrar columnas
weather = weather.rename(columns={
    "wind_speed_ms": "wind_speed",
    "temperature_C": "temperature",
    "pressure_Pa": "pressure"
})


# Limpieza de datos
for col in weather.columns:
    weather[col] = weather[col].astype(str).str.replace(",", ".").astype(float)

weather = weather.dropna(axis=1, how='all')


# Añadir roughness_length
weather['roughness_length'] = 0.03


# === MultiIndex con alturas ===
heights = {
    'wind_speed': 10,
    'temperature': 2,
    'pressure': 0,
    'roughness_length': 0
}

column_arrays = [list(weather.columns), [heights[col] for col in weather.columns]]
weather.columns = pd.MultiIndex.from_arrays(column_arrays)


# === 3. Simulación ===
resultados = {}


# Turbinas de librería
for nombre, config in turbinas_config.items():

    turbina = WindTurbine(**config)

    mc = ModelChain(
        turbina,
        wind_speed_model='logarithmic',
        density_model='ideal_gas',
        temperature_model='linear_gradient',
        power_output_model='power_curve',
        density_correction=True,
        obstacle_height=0,
        roughness_length=0.03
    )

    mc.run_model(weather)

    df = pd.DataFrame({
        'potencia_w': mc.power_output,
        'velocidad_viento': weather[('wind_speed', 10)]
    })

    df.index = pd.to_datetime(df.index)

    tiempos_horas = (df.index - df.index[0]).total_seconds() / 3600.0
    energia_wh = np.trapz(df['potencia_w'].values, tiempos_horas)

    energia_kwh = energia_wh / 1000.0
    energia_mwh = energia_kwh / 1000.0

    duracion_horas = (df.index[-1] - df.index[0]).total_seconds() / 3600.0
    energia_max = (turbina.nominal_power * duracion_horas) / 1000.0

    factor_cap = (energia_kwh / energia_max) * 100.0

    resultados[nombre] = {
        "Energía (MWh)": energia_mwh,
        "Factor capacidad (%)": factor_cap,
        "Potencia promedio (kW)": df['potencia_w'].mean() / 1000,
        "Nominal (kW)": turbina.nominal_power / 1000
    }


# Mi turbina personalizada
mc = ModelChain(
    mi_turbina,
    wind_speed_model='logarithmic',
    density_model='ideal_gas',
    temperature_model='linear_gradient',
    power_output_model='power_curve',
    density_correction=True,
    obstacle_height=0,
    roughness_length=0.03
)

mc.run_model(weather)

df = pd.DataFrame({
    'potencia_w': mc.power_output,
    'velocidad_viento': weather[('wind_speed', 10)]
})

df.index = pd.to_datetime(df.index)

tiempos_horas = (df.index - df.index[0]).total_seconds() / 3600.0
energia_wh = np.trapz(df['potencia_w'].values, tiempos_horas)

energia_kwh = energia_wh / 1000.0
energia_mwh = energia_kwh / 1000.0

duracion_horas = (df.index[-1] - df.index[0]).total_seconds() / 3600.0
energia_max = (mi_turbina.nominal_power * duracion_horas) / 1000.0

factor_cap = (energia_kwh / energia_max) * 100.0

resultados["MiTurbina/800kW"] = {
    "Energía (MWh)": energia_mwh,
    "Factor capacidad (%)": factor_cap,
    "Potencia promedio (kW)": df['potencia_w'].mean() / 1000,
    "Nominal (kW)": mi_turbina.nominal_power / 1000
}


# === 4. Mostrar resultados comparativos ===
print("\n" + "="*60)
print("RESUMEN COMPARATIVO")
print("="*60)

for nombre, res in resultados.items():
    print(f"\nTurbina: {nombre}")
    for k, v in res.items():
        print(f"  {k}: {v:.2f}")


# === 5. Graficar curva de potencia solo de tu turbina ===
pc = mi_turbina.power_curve

plt.figure(figsize=(8,5))
plt.plot(pc['wind_speed'], pc['value'] / 1000, marker="o", color="g")

plt.title("Curva de Potencia - MiTurbina 800kW")
plt.xlabel("Velocidad del viento [m/s]")
plt.ylabel("Potencia [kW]")

plt.grid(True)
plt.show()


# Resumen: Este código compara 4 tipos de turbinas, 3 reales que se desaprovechan debido a las bajas velocidades de viento y una turbina
# escogida a conveniencia que sí aprovecha las velocidades bajas de viento de La Guajira