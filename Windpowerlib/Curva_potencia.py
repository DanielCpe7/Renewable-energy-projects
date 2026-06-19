#En este codigo es posible obtener la curva de potencia de cualquiera de los 66 modelos de turbinas que vienen por defecto con la libreria
import matplotlib.pyplot as plt
from windpowerlib import WindTurbine

# Crear la turbina usando la base de datos interna
turbine = WindTurbine(turbine_type="E48/800", hub_height=60)

# Extraer la curva de potencia
power_curve = turbine.power_curve  # DataFrame con columnas: wind_speed, power

print(power_curve.head())  # Para ver las primeras filas

# Graficar
plt.figure(figsize=(8,5))
plt.plot(power_curve['wind_speed'], power_curve['value'], marker="o", color="b")
plt.title("Curva de Potencia -Enercon    E48/800")
plt.xlabel("Velocidad del viento [m/s]")
plt.ylabel("Potencia [W]")
plt.grid(True)
plt.show()
