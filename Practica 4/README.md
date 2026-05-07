# Práctica 4 – De OpenMP a MPI: Integración numérica por el método del trapecio

## Descripción

Contiene dos implementaciones del cálculo de la integral definida de  
**f(x) = x³/3 + 4x** usando la regla del trapecio compuesta:

| Archivo | Modelo de paralelismo |
|---|---|
| `integral.c` | OpenMP (memoria compartida) |
| `integral_mpi.c` | MPI (memoria distribuida) |

---

## Compilación

### OpenMP
```bash
gcc -O2 -fopenmp -o integral integral.c
```

### MPI
```bash
mpicc -O2 -o integral_mpi integral_mpi.c
```

---

## Ejecución

### OpenMP (límites a=0, b=10, n=1000 trapezoides, 4 hilos)
```bash
OMP_NUM_THREADS=4 ./integral 0 10 1000
```

### MPI (4 procesos, mismos parámetros)
```bash
mpirun -np 4 ./integral_mpi 0 10 1000
```

---

## Transformación: de OpenMP a MPI

La lógica matemática es idéntica en ambas versiones: se divide el intervalo
`[a, b]` en `n` subintervalos de ancho `h = (b-a)/n` y cada unidad de
ejecución (hilo u proceso) calcula la suma parcial de su rango local, que
luego se acumula en el resultado final.

Los cambios estructurales son los siguientes:

| Aspecto | OpenMP (`integral.c`) | MPI (`integral_mpi.c`) |
|---|---|---|
| División del trabajo | `omp_get_thread_num()` + `omp_get_num_threads()` | `MPI_Comm_rank` + `MPI_Comm_size` |
| Distribución de parámetros | Variables compartidas en memoria | `MPI_Bcast` desde el proceso raíz |
| Acumulación del resultado | `#pragma omp critical` sobre `total` | `MPI_Reduce` con `MPI_SUM` |
| Sincronización | Implícita al salir del bloque `parallel` | Implícita en `MPI_Reduce` |
| Bucle interno | `#pragma omp parallel for ordered` (anidado) | Bucle C estándar por proceso |

> **Nota sobre el código OpenMP original:** el uso de `#pragma omp parallel for ordered`
> dentro de una región `parallel` ya existente genera trabajo anidado innecesario y
> serializa el bucle con la cláusula `ordered`. En la versión MPI este patrón desaparece
> porque cada proceso ejecuta su propio bucle de forma completamente independiente.

---

## Análisis: ventajas y desventajas

### MPI (`integral_mpi.c`)

#### Ventajas

| # | Ventaja | Descripción |
|---|---|---|
| 1 | **Escalabilidad distribuida** | Puede ejecutarse en un clúster con decenas o cientos de nodos; no está limitado por los núcleos de una sola máquina. |
| 2 | **Sin condiciones de carrera** | Cada proceso tiene su propio espacio de memoria; no existen variables compartidas que proteger. El resultado parcial se combina una única vez mediante `MPI_Reduce`. |
| 3 | **Portabilidad** | El estándar MPI está disponible en Linux, macOS y Windows (OpenMPI, MPICH, Intel MPI). El mismo binario puede correr en 1 o en 1000 procesos sin recompilar. |
| 4 | **Tolerancia al fallo (avanzado)** | Herramientas como ULFM-MPI permiten continuar el cómputo aunque un nodo falle; algo imposible con hilos OpenMP. |

#### Desventajas

| # | Desventaja | Descripción |
|---|---|---|
| 1 | **Sobrecarga de comunicación** | `MPI_Bcast` y `MPI_Reduce` tienen latencia de red. Para integrales simples (poco trabajo por proceso), la comunicación puede dominar sobre el cómputo. |
| 2 | **Mayor verbosidad del código** | Se necesitan llamadas explícitas a `MPI_Init`, `MPI_Bcast`, `MPI_Reduce` y `MPI_Finalize`; el código es más largo y difícil de mantener. |
| 3 | **Distribución de datos manual** | Si `n` no es divisible exactamente entre el número de procesos, los trapezoides sobrantes se pierden. En el código OpenMP ocurre el mismo problema, pero es más fácil de corregir con un bloque `critical`. |
| 4 | **Inicialización costosa** | Lanzar múltiples procesos MPI tiene mayor costo inicial que crear hilos dentro de un proceso ya en ejecución. |

---

### OpenMP (`integral.c`)

#### Ventajas

| # | Ventaja | Descripción |
|---|---|---|
| 1 | **Sencillez de implementación** | Basta con añadir directivas `#pragma omp` al código secuencial. La curva de aprendizaje es mucho menor. |
| 2 | **Baja latencia** | Los hilos comparten memoria; no hay serialización de mensajes ni latencia de red. |
| 3 | **Menor verbosidad** | El código fuente es casi idéntico al secuencial. |

#### Desventajas

| # | Desventaja | Descripción |
|---|---|---|
| 1 | **Limitado a un nodo** | Solo usa los núcleos del procesador local; no puede aprovechar un clúster. |
| 2 | **Condiciones de carrera** | Accesos a variables compartidas como `total` requieren secciones críticas (`#pragma omp critical`), lo que serializa parte del trabajo. |
| 3 | **Paralelismo anidado problemático** | El código original usa `parallel for ordered` dentro de una región `parallel`, lo que anula la ganancia del bucle interno y añade complejidad sin beneficio. |

---

## Conclusión

Para una integral numéricamente simple como ésta, **OpenMP es preferible en una sola
máquina** por su menor sobrecarga. **MPI es la elección correcta cuando el dominio es
demasiado grande para un único nodo** o cuando se necesita escalar horizontalmente en
un clúster. En la práctica, ambos modelos pueden combinarse (MPI + OpenMP híbrido)
para aprovechar al máximo tanto la memoria distribuida entre nodos como la memoria
compartida dentro de cada nodo.
