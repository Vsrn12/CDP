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