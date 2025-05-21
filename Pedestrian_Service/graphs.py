import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Estilo de gráficos
sns.set(style="whitegrid")

# Crear carpeta de salida si no existe
os.makedirs("graficas", exist_ok=True)

# Cargar el CSV
try:
    df = pd.read_csv("detecciones_5.csv", parse_dates=["datetime"])
except Exception as e:
    print(f"❌ Error al cargar CSV: {e}")
    exit()

# Asegurar tipos correctos
df["region"] = df["region"].astype(str)
df["person_id"] = df["person_id"].astype(str)

# ---------- Gráfico 1: Conteo total por región ----------
plt.figure(figsize=(10, 6))
sns.countplot(x="region", data=df, palette="viridis")
plt.title("Cantidad total de detecciones por región")
plt.xlabel("Región")
plt.ylabel("Cantidad de detecciones")
plt.tight_layout()
plt.savefig("graficas/detecciones_por_region_5.png")
plt.close()

# ---------- Gráfico 2: Personas únicas por región ----------
personas_unicas = df.groupby("region")["person_id"].nunique().reset_index()
personas_unicas.columns = ["region", "personas_unicas"]

plt.figure(figsize=(10, 6))
sns.barplot(data=personas_unicas, x="region", y="personas_unicas", palette="magma")
plt.title("Cantidad de personas únicas por región")
plt.xlabel("Región")
plt.ylabel("Personas únicas")
plt.tight_layout()
plt.savefig("graficas/personas_unicas_por_region_5.png")
plt.close()

# ---------- Gráfico 3: Detecciones por hora ----------
df["hora"] = df["datetime"].dt.hour

plt.figure(figsize=(10, 6))
sns.countplot(x="hora", data=df, palette="coolwarm")
plt.title("Detecciones por hora del día")
plt.xlabel("Hora")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.savefig("graficas/detecciones_por_hora_5.png")
plt.close()

# ---------- Gráfico 4: Detecciones en el tiempo ----------
df["minute"] = df["datetime"].dt.floor("min")
serie_tiempo = df.groupby("minute").size().reset_index(name="conteo")

plt.figure(figsize=(12, 6))
sns.lineplot(x="minute", y="conteo", data=serie_tiempo, marker="o", color="teal")
plt.title("Evolución de detecciones por minuto")
plt.xlabel("Fecha y hora")
plt.ylabel("Detecciones")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graficas/evolucion_detecciones_5.png")
plt.close()

print("✅ Gráficas generadas y guardadas en la carpeta 'graficas'")
