# Paralelizacion de la serie de Leibniz con OpenMP

## Cambio realizado

El programa fue modificado para que el calculo principal use un `parallel for` de OpenMP en lugar de repartir manualmente las iteraciones entre hilos.

La directiva utilizada es:

```c
#pragma omp parallel for reduction(+:respuesta) schedule(static)
```

Con esto:

- El `for` se ejecuta en paralelo.
- Cada hilo acumula una copia privada de `respuesta`.
- OpenMP combina automaticamente los resultados al final mediante `reduction`.

## Compilacion

Con GCC:

```bash
gcc -fopenmp pi_leibniz_openmp.c -o pi_openmp
```

## Ejecucion

```bash
./pi_openmp
```

## Ejemplo de funcionamiento

Si se ingresan muchas iteraciones, el valor calculado se aproxima a `pi`.