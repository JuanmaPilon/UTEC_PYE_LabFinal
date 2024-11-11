import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Se usa numpy para utilizar biblioteca nativa de Py para calculos
# Se usa scipy, especificamente scipy.stats para poder abordar la parte de estadistica y probabilidad
# Se usa matplotlib para podes graficar todo lo que necesitemos

# Se deben tener los 3 modulos o biblitecas para poder correr el programa correctamente
# - pip install numpy
# - pip install scipy
# - pip install matplotlib

# Todos los parametros iniciales
n_usuarios = 100          # Valor de numero de usuarios por defecto
k_cajas = 3               # Valor de cajas entre 1 y 5, fijado por defecto en k = 3
mu_llegadas = 3           # Media de tiempo entre llegadas (Distribucion Poisson)
mu_productos = 5          # Media de cantidad de productos (Distribucion Normal)
sigma_productos = 3       # Desviacion estandar de cantidad de productos
p_efectivo = 0.4          # Probabilidad de pagar en efectivo
tiempo_pago_efectivo = 2  # Tiempo de pago en efectivo (minutos)
tiempo_pago_otro = 70 / 60  # Tiempo de pago en otro medio (minutos)


# Esta funcion  genera un tiempo aleatorio entre la llegada de clientes utilizando una distribucion de Poisson con una media de mu_llegadas (3)
# de aqui sacamos los eventos en intervalos especificos
def generar_tiempo_llegada():
    return stats.poisson(mu_llegadas).rvs()

def generar_tiempo_uso_caja():
    # Generar tiempo basado en la cantidad de productos
    tiempo_productos = max(0, stats.norm(mu_productos, sigma_productos).rvs())
    
    # Determinar el tiempo de pago segun sea el tipo de pago
    pago_efectivo = stats.bernoulli(p_efectivo).rvs()
    tiempo_pago = tiempo_pago_efectivo if pago_efectivo == 1 else tiempo_pago_otro
    
    # Tiempo total de uso de la caja
    # Suma el valor de dist normal + el valor que depende del pago dist Bernoulli
    return tiempo_productos + tiempo_pago

# Fila unica
def simulacion_fila_unica(n_usuarios, k_cajas):
    tiempo_uso_cajas = [[] for _ in range(k_cajas)]
    tiempo_espera = []
    llegada_actual = 0
    tiempos_finales_cajas = np.zeros(k_cajas)
    
    for _ in range(n_usuarios):
        llegada_actual += generar_tiempo_llegada()
        tiempo_uso = generar_tiempo_uso_caja()
        
        # Seleccionar la primera caja disponible
        caja_elegida = np.argmin(tiempos_finales_cajas)
        
        # Calcular el tiempo de espera en la fila
        tiempo_inicio = max(llegada_actual, tiempos_finales_cajas[caja_elegida])
        tiempo_espera_cliente = tiempo_inicio - llegada_actual
        tiempo_espera.append(tiempo_espera_cliente)
        
        # Actualizar tiempos
        tiempos_finales_cajas[caja_elegida] = tiempo_inicio + tiempo_uso
        tiempo_uso_cajas[caja_elegida].append(tiempo_uso)
    
    return tiempo_uso_cajas, tiempo_espera

# Varias filas
def simulacion_filas_independientes(n_usuarios, k_cajas):
    tiempo_uso_cajas = [[] for _ in range(k_cajas)]
    tiempo_espera = []
    llegada_actual = 0
    tiempos_finales_cajas = np.zeros(k_cajas)
    
    for _ in range(n_usuarios):
        llegada_actual += generar_tiempo_llegada()
        tiempo_uso = generar_tiempo_uso_caja()
        
        # Seleccionar la caja con menos clientes en fila
        caja_elegida = np.argmin([len(caja) for caja in tiempo_uso_cajas])
        
        # Calcular el tiempo de espera en la fila
        tiempo_inicio = max(llegada_actual, tiempos_finales_cajas[caja_elegida])
        tiempo_espera_cliente = tiempo_inicio - llegada_actual
        tiempo_espera.append(tiempo_espera_cliente)
        
        # Actualizar tiempos
        tiempos_finales_cajas[caja_elegida] = tiempo_inicio + tiempo_uso
        tiempo_uso_cajas[caja_elegida].append(tiempo_uso)
    
    return tiempo_uso_cajas, tiempo_espera

# Analizar y graficar
def analizar_resultados(tiempo_uso_cajas, tiempo_espera):
    # Calcular estadísticas para cada caja
    tiempo_uso_stats = [(np.mean(caja), np.std(caja)) for caja in tiempo_uso_cajas]
    tiempo_espera_stats = (np.mean(tiempo_espera), np.std(tiempo_espera))
    
    # Graficar tiempo de uso de cada caja
    plt.figure(figsize=(10, 5))
    for idx, caja in enumerate(tiempo_uso_cajas):
        plt.plot(caja, label=f'Caja {idx+1}')
    plt.xlabel("Cliente")
    plt.ylabel("Tiempo de uso (minutos)")
    plt.title("Tiempo de uso en cada caja")
    plt.legend()
    plt.show()
    
    # Graficar tiempo de espera de cada cliente
    plt.figure(figsize=(10, 5))
    plt.plot(tiempo_espera, label="Tiempo de espera")
    plt.xlabel("Cliente")
    plt.ylabel("Tiempo de espera (minutos)")
    plt.title("Tiempo de espera de los clientes")
    plt.legend()
    plt.show()
    
    return tiempo_uso_stats, tiempo_espera_stats

# Ejecución

# Simulación con Fila Única
uso_cajas_fila_unica, espera_fila_unica = simulacion_fila_unica(n_usuarios, k_cajas)
uso_stats_fila_unica, espera_stats_fila_unica = analizar_resultados(uso_cajas_fila_unica, espera_fila_unica)

# Simulación con Filas Independientes
uso_cajas_filas_ind, espera_filas_ind = simulacion_filas_independientes(n_usuarios, k_cajas)
uso_stats_filas_ind, espera_stats_filas_ind = analizar_resultados(uso_cajas_filas_ind, espera_filas_ind)
