# Práctica 3 – Comparativa MPI vs OpenMP para cálculo de promedio

## Descripción

Este repositorio contiene dos implementaciones para calcular el promedio de un arreglo de números flotantes aleatorios en paralelo:

| Archivo | Modelo de paralelismo |
|---|---|
| `avg.c` | MPI (Message Passing Interface) |
| `avg_omp.c` | OpenMP (memoria compartida) |

---

## Compilación

### MPI
```bash
mpicc -O2 -o avg avg.c
```

### OpenMP
```bash
gcc -O2 -fopenmp -o avg_omp avg_omp.c
```

---

## Ejecución

### MPI (4 procesos, 1 000 000 elementos por proceso → 4 000 000 en total)
```bash
mpirun -np 4 ./avg 1000000
```

### OpenMP (4 000 000 elementos, 4 hilos)
```bash
./avg_omp 4000000 4
```

---

## Directiva `reduction` en cada modelo

### MPI
En MPI no existe una directiva `reduction` como palabra clave del lenguaje; la reducción se implementa **explícitamente** mediante `MPI_Reduce` o, en este caso, con `MPI_Scatter` + `MPI_Gather` + cálculo manual del promedio de promedios parciales:

```c
// Cada proceso calcula su promedio parcial
float sub_avg = compute_avg(sub_rand_nums, num_elements_per_proc);

// Se reúnen en el proceso raíz
MPI_Gather(&sub_avg, 1, MPI_FLOAT, sub_avgs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

// El proceso raíz calcula el promedio final
float avg = compute_avg(sub_avgs, world_size);
```

### OpenMP
La directiva `reduction` es nativa de OpenMP y se declara directamente en el pragma. Cada hilo mantiene una **copia privada** de `sum`; al terminar el bucle, todas las copias se combinan (suman) en la variable compartida original:

```c
// Cada hilo acumula una copia privada de 'sum'; al finalizar el bucle,
// todas las copias se suman en la variable compartida original.
#pragma omp parallel for reduction(+:sum)
for (i = 0; i < num_elements; i++) {
    sum += array[i];
}
```

---

## Comparación de ejecuciones

### Resultados esperados (hardware de referencia: 4 núcleos, 1 000 000 elementos)

| Métrica | MPI (4 procesos) | OpenMP (4 hilos) |
|---|---|---|
| Modelo | Memoria distribuida | Memoria compartida |
| Tiempo de inicialización | Alto (lanzar procesos + MPI_Init) | Bajo (los hilos comparten el proceso) |
| Tiempo de cómputo | Similar | Similar |
| Overhead de comunicación | MPI_Scatter + MPI_Gather | Ninguno (acceso directo a memoria) |
| Escalabilidad a múltiples nodos | Sí | No (limitado a un nodo) |
| Resultado numérico | Igual (flotante, pequeñas diferencias de redondeo) | Igual |

> **Nota:** para mediciones reales ejecute ambos programas con la misma semilla o un número fijo de elementos y tome tiempos con `MPI_Wtime` (MPI) y `omp_get_wtime` (OpenMP).

---

## Ventajas y desventajas de cada modelo de programación paralela

### MPI (Message Passing Interface)

| Ventajas | Desventajas |
|---|---|
| Escala a cientos/miles de nodos en clústeres | Programación más compleja: el programador gestiona explícitamente el envío y recepción de datos |
| Sin condiciones de carrera: cada proceso tiene su propio espacio de memoria | Mayor overhead de comunicación (latencia de red o bus entre procesos) |
| Portabilidad total: funciona en sistemas distribuidos y de memoria compartida | Mayor uso de memoria: cada proceso duplica los datos que necesita |
| Modelo de fallos más robusto en aplicaciones distribuidas | Difícil de depurar (mensajes, deadlocks, desbalance de carga) |
| Bien adaptado a problemas con partición natural de datos | Tiempo de arranque elevado (`MPI_Init`, lanzamiento de procesos) |

### OpenMP (Open Multi-Processing)

| Ventajas | Desventajas |
|---|---|
| Programación incremental: se añaden pragmas al código secuencial | Limitado a memoria compartida (un solo nodo) |
| Bajo overhead de creación de hilos (comparado con procesos MPI) | Condiciones de carrera si se accede a variables compartidas sin sincronización |
| La directiva `reduction` simplifica la acumulación paralela | El rendimiento depende del número de núcleos disponibles en la máquina |
| Código más legible y mantenible para paralelismo de bucles | Falsa compartición de caché (*false sharing*) puede degradar el rendimiento |
| Ideal para paralelismo de grano fino dentro de un nodo | Menor escalabilidad en comparación con MPI para grandes clústeres |

### Modelo híbrido MPI + OpenMP

Una práctica común en HPC es combinar ambos modelos:
- **MPI** entre nodos para distribuir datos.
- **OpenMP** dentro de cada nodo para aprovechar múltiples núcleos.

Esto combina la escalabilidad de MPI con la eficiencia de memoria de OpenMP.

---

## Conclusión

- Usa **OpenMP** cuando el problema cabe en la memoria de una sola máquina y quieres paralelismo sencillo con pocos cambios al código secuencial.
- Usa **MPI** cuando necesitas escalar más allá de un nodo o cuando trabajas en arquitecturas de memoria distribuida.
- Usa **MPI + OpenMP** para obtener lo mejor de ambos mundos en clústeres modernos de múltiples núcleos.
