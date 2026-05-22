# Implementaciones Paralelas: OpenMP, MPI y CUDA

## Estructura del Repositorio

```
.
├── pi_leibniz.c            # PI - Serie de Leibniz (OpenMP - ORIGINAL)
├── pi_leibniz_mpi.c        # PI - Serie de Leibniz (MPI)
├── pi_leibniz_cuda.cu      # PI - Serie de Leibniz (CUDA)
│
├── pi_montecarlo.c         # PI - Monte Carlo (OpenMP - ORIGINAL)
├── pi_montecarlo_mpi.c     # PI - Monte Carlo (MPI)
├── pi_montecarlo_cuda.cu   # PI - Monte Carlo (CUDA)
│
├── reduce_avg.c            # Promedio con reducción (MPI - ORIGINAL)
├── reduce_avg_omp.c        # Promedio con reducción (OpenMP)
├── reduce_avg_cuda.cu      # Promedio con reducción (CUDA)
│
├── reduce_stddev.c         # Desviación estándar (MPI - ORIGINAL)
├── reduce_stddev_omp.c     # Desviación estándar (OpenMP)
└── reduce_stddev_cuda.cu   # Desviación estándar (CUDA)
```

---

## Compilación y Ejecución

### Serie de Leibniz

| Versión | Compilar | Ejecutar |
|---------|----------|----------|
| OpenMP  | `gcc -O2 -fopenmp -o pi_leibniz pi_leibniz.c` | `./pi_leibniz` |
| MPI     | `mpicc -O2 -o pi_leibniz_mpi pi_leibniz_mpi.c` | `mpirun -np 4 ./pi_leibniz_mpi` |
| CUDA    | `nvcc -O2 -o pi_leibniz_cuda pi_leibniz_cuda.cu` | `./pi_leibniz_cuda` |

### Monte Carlo

| Versión | Compilar | Ejecutar |
|---------|----------|----------|
| OpenMP  | `gcc -O2 -fopenmp -o pi_montecarlo pi_montecarlo.c` | `./pi_montecarlo <samples>` |
| MPI     | `mpicc -O2 -o pi_montecarlo_mpi pi_montecarlo_mpi.c` | `mpirun -np 4 ./pi_montecarlo_mpi <samples>` |
| CUDA    | `nvcc -O2 -lcurand -o pi_montecarlo_cuda pi_montecarlo_cuda.cu` | `./pi_montecarlo_cuda <samples>` |

### Promedio (reduce_avg)

| Versión | Compilar | Ejecutar |
|---------|----------|----------|
| MPI     | `mpicc -O2 -o reduce_avg reduce_avg.c` | `mpirun -np 4 ./reduce_avg <n>` |
| OpenMP  | `gcc -O2 -fopenmp -o reduce_avg_omp reduce_avg_omp.c` | `./reduce_avg_omp <n>` |
| CUDA    | `nvcc -O2 -o reduce_avg_cuda reduce_avg_cuda.cu` | `./reduce_avg_cuda <n>` |

### Desviación Estándar (reduce_stddev)

| Versión | Compilar | Ejecutar |
|---------|----------|----------|
| MPI     | `mpicc -O2 -lm -o reduce_stddev reduce_stddev.c` | `mpirun -np 4 ./reduce_stddev <n>` |
| OpenMP  | `gcc -O2 -fopenmp -lm -o reduce_stddev_omp reduce_stddev_omp.c` | `./reduce_stddev_omp <n>` |
| CUDA    | `nvcc -O2 -o reduce_stddev_cuda reduce_stddev_cuda.cu` | `./reduce_stddev_cuda <n>` |

---

## Descripción de los Algoritmos

### 1. Serie de Leibniz

La fórmula de Leibniz-Madhava converge lentamente a π:

$$\pi = 4 \sum_{n=0}^{N} \frac{(-1)^n}{2n+1} = 4\left(1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \cdots\right)$$

El paralelismo natural es la **distribución cíclica de iteraciones**: el índice `n` se reparte entre los trabajadores de forma `n = id, id+P, id+2P, ...`, donde `P` es el número total de trabajadores. Cada trabajador acumula su suma parcial y al final se reducen todas en un único resultado.

### 2. Monte Carlo para π

Se generan `N` puntos uniformes $(x, y) \in [0,1]^2$. Si $x^2 + y^2 \leq 1$, el punto cae dentro del cuarto de círculo. La estimación es:

$$\pi \approx 4 \cdot \frac{\text{puntos dentro del círculo}}{N}$$

El paralelismo se logra dividiendo los `N` puntos entre los trabajadores, cada uno con su propio generador de números aleatorios (semilla distinta). Al final se suman los conteos locales.

### 3. Reducción para promedio

Cada trabajador genera y suma un subconjunto del arreglo (suma local). La suma global se obtiene combinando todas las sumas locales mediante una operación de reducción. El promedio es `suma_global / N_total`.

### 4. Reducción para desviación estándar

Requiere **dos pasadas**:
1. **Pasada 1 (media):** reducción de la suma → `mean = sum / N`.
2. **Pasada 2 (varianza):** cada trabajador calcula $\sum (x_i - \bar{x})^2$ sobre sus datos. La suma de cuadráticas global se reduce y la desviación estándar es:

$$\sigma = \sqrt{\frac{\sum (x_i - \bar{x})^2}{N}}$$

---

## Análisis Comparativo de Implementaciones

### OpenMP

**Modelo:** Memoria compartida. Múltiples hilos dentro de un mismo proceso y nodo.

#### Ventajas
- **Portabilidad y simplicidad:** se agrega paralelismo con directivas `#pragma omp` mínimas; el código base apenas cambia.
- **Sin overhead de comunicación:** los hilos comparten el espacio de memoria del proceso; no hay transferencia de datos entre nodos.
- **Sincronización automática:** las cláusulas `reduction`, `shared` y `private` gestionan la coherencia sin código explícito de sincronización.
- **Arranque rápido:** el coste de crear/destruir hilos es muy bajo comparado con el lanzamiento de procesos MPI.
- **Escalabilidad implícita:** `omp_get_max_threads()` detecta los núcleos disponibles automáticamente.

#### Desventajas
- **Limitado a un nodo:** solo escala hasta los núcleos de una máquina; no puede aprovechar clústeres.
- **Contención de memoria (false sharing):** si dos hilos acceden a líneas de caché contiguas, el rendimiento cae aunque los datos sean independientes.
- **Depuración compleja:** condiciones de carrera y deadlocks son difíciles de reproducir.
- **Speedup limitado por Ley de Amdahl:** las secciones seriales (entrada de datos, acumulación final) imponen un techo.

---

### MPI

**Modelo:** Memoria distribuida. Múltiples procesos independientes que se comunican por paso de mensajes.

#### Ventajas
- **Escalabilidad horizontal:** puede usar cientos o miles de nodos en un clúster; no hay límite de memoria compartida.
- **Aislamiento de procesos:** cada proceso tiene su propio espacio de memoria; no hay condiciones de carrera por defecto.
- **Portabilidad en HPC:** es el estándar de facto en supercomputadoras; funciona en prácticamente cualquier arquitectura.
- **Control fino del flujo de datos:** `MPI_Bcast`, `MPI_Reduce`, `MPI_Allreduce`, `MPI_Scatter`/`Gather` permiten patrones de comunicación precisos.

#### Desventajas
- **Mayor verbosidad:** requiere código explícito de inicialización, comunicación y finalización (`MPI_Init`, `MPI_Comm_rank`, etc.).
- **Overhead de comunicación:** el intercambio de mensajes entre procesos (especialmente entre nodos) puede dominar el tiempo cuando la carga de cómputo es pequeña.
- **Gestión manual de datos:** el programador debe diseñar la distribución del trabajo y asegurarse de que cada proceso tenga los datos que necesita.
- **Latencia de arranque:** lanzar muchos procesos MPI tiene un coste inicial mayor que crear hilos OpenMP.
- **Debugging difícil en paralelo distribuido:** los errores de comunicación (deadlocks, mensajes perdidos) son complicados de diagnosticar.

---

### CUDA

**Modelo:** Heterogéneo CPU+GPU. Miles de hilos ligeros ejecutándose en una GPU (SIMT).

#### Ventajas
- **Throughput masivo:** una GPU moderna tiene miles de núcleos CUDA ejecutando en paralelo; ideal para operaciones altamente regulares (sumas, reducciones, Monte Carlo).
- **Ancho de banda de memoria elevado:** la memoria de la GPU (HBM/GDDR) ofrece cientos de GB/s, muy superiores a la RAM del CPU.
- **Reducción en memoria compartida:** el patrón de reducción logarítmica en `__shared__` minimiza los accesos a memoria global.
- **Librerías optimizadas:** cuRAND (números aleatorios), cuBLAS, Thrust, etc., facilitan implementaciones de alto rendimiento sin escribir kernels desde cero.

#### Desventajas
- **Overhead de transferencia PCI-e:** copiar datos entre CPU y GPU (`cudaMemcpy`) puede ser el cuello de botella para conjuntos de datos pequeños.
- **Requiere hardware especializado:** necesita una GPU NVIDIA compatible con CUDA; no es portable a CPUs o GPUs de otros fabricantes sin reescribir el código.
- **Curva de aprendizaje elevada:** el programador debe entender la jerarquía de memoria (global, shared, registers), la organización de bloques/hilos y la divergencia de warp.
- **Sincronización entre bloques limitada:** `__syncthreads()` solo sincroniza dentro de un bloque; la reducción final entre bloques debe hacerse en CPU o con kernels adicionales.
- **Depuración compleja:** herramientas como `cuda-gdb` y `compute-sanitizer` son necesarias pero tienen limitaciones respecto a GDB estándar.

---

## Tabla Resumen de Comparación

| Criterio | OpenMP | MPI | CUDA |
|---|---|---|---|
| **Modelo** | Mem. compartida | Mem. distribuida | GPU (SIMT) |
| **Escalabilidad** | 1 nodo | Multinodo | 1 GPU |
| **Facilidad de programación** | Alta | Media | Baja |
| **Overhead de comunicación** | Mínimo | Medio–Alto | PCI-e latencia |
| **Paralelismo máximo** | ~64 hilos (típico) | Miles de procesos | Miles de hilos |
| **Caso ideal** | Nodos multicore | Clústeres HPC | Operaciones regulares masivas |
| **Portabilidad** | Alta (cualquier CPU) | Alta (cualquier plataforma) | Baja (solo NVIDIA) |
| **Herramientas de debug** | Valgrind, Helgrind | TotalView, DDT | cuda-gdb, sanitizer |

---

## Consideraciones por Algoritmo

### Serie de Leibniz

- **OpenMP** es el más sencillo: la distribución cíclica se expresa naturalmente con `#pragma omp parallel` y sumas parciales por hilo.
- **MPI** distribuye bloques cíclicos entre procesos; el `MPI_Reduce` final es eficiente aunque hay latencia por la llamada colectiva.
- **CUDA** es la más rápida para `N` muy grande (> 10⁸): miles de hilos recorren la serie en paralelo. Para `N` pequeño, el overhead de lanzar el kernel supera el beneficio.

### Monte Carlo

- **OpenMP** aprovecha `erand48` con semillas por hilo; la cláusula `reduction(+:count)` elimina condiciones de carrera en el conteo.
- **MPI** es natural: cada proceso genera puntos independientes; la comunicación final es un único `MPI_Reduce`.
- **CUDA** necesita cuRAND para generar números aleatorios en el dispositivo; es la versión más rápida para `N` > 10⁷, pero cuRAND tiene un coste de inicialización por hilo (`curand_init`) que puede ser significativo.

### Promedio y Desviación Estándar

- **OpenMP** es trivial: una línea de reducción paralela. Ideal para un único nodo con grandes arreglos en memoria.
- **MPI** distribuye el arreglo entre procesos; para `reduce_stddev` se necesita `MPI_Allreduce` (la media debe conocerla todos los procesos antes del paso 2).
- **CUDA** usa el patrón clásico de reducción en shared memory (dos kernels para stddev). Es la opción más rápida para arreglos de decenas de millones de elementos que caben en la VRAM.
