from windpowerlib import WindTurbine, ModelChain
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 1. Definir turbina personalizada ===
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
    # Datos de ejemplo si falla la lectura
    data = {
        'wind_speed_ms': [3.2, 3.8, 4.1, 3.5, 4.3, 4.8, 5.0, 4.5, 3.9, 4.2],
        'temperature_C': [29.0, 29.6, 30.2, 31.0, 30.5, 31.8, 32.0, 31.2, 30.1, 30.4],
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

# Reemplazar comas por puntos y convertir a float
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

# === 3. Simulación para una turbina ===
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

# === 4. Calcular energía ===
tiempos_horas = (df.index - df.index[0]).total_seconds() / 3600.0
energia_wh = np.trapz(df['potencia_w'].values, tiempos_horas)
energia_kwh = energia_wh / 1000.0
energia_mwh = energia_kwh / 1000.0

# === 5. Multiplicar por número de turbinas ===
n_turbinas = 10
energia_mwh_total = energia_mwh * n_turbinas
potencia_promedio_kw = (df['potencia_w'].mean() / 1000) * n_turbinas

# === 6. Calcular factor de capacidad ===
duracion_horas = (df.index[-1] - df.index[0]).total_seconds() / 3600.0
energia_max = (mi_turbina.nominal_power * duracion_horas * n_turbinas) / 1000.0  # en kWh
factor_cap = (energia_kwh * n_turbinas / energia_max) * 100.0

# === 7. Mostrar resultados ===
print("\n" + "="*60)
print(f"RESULTADOS - Parque Eólico con {n_turbinas} turbinas")
print("="*60)
print("Turbina usada: MiTurbina/800")
print(f"Energía total generada: {energia_mwh_total:.3f} MWh")
print(f"Potencia promedio total: {potencia_promedio_kw:.2f} kW")
print(f"Potencia nominal total: {mi_turbina.nominal_power * n_turbinas / 1000:.0f} kW")
print(f"Factor de capacidad: {factor_cap:.2f} %")

# === 8. Graficar curva de potencia ===
pc = mi_turbina.power_curve
plt.figure(figsize=(8,5))
plt.plot(pc['wind_speed'], pc['value']/1000, marker="o", color="g")
plt.title("Curva de Potencia - MiTurbina 800kW")
plt.xlabel("Velocidad del viento [m/s]")
plt.ylabel("Potencia [kW]")
plt.grid(True)
plt.show()