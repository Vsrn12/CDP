/*
 * reduce_stddev_omp.c
 * Cálculo de la desviación estándar de un arreglo de números aleatorios
 * usando OpenMP.
 * Paso 1: reducción paralela para obtener la media global.
 * Paso 2: reducción paralela para la suma de diferencias cuadráticas.
 *
 * Compilar: gcc -O2 -fopenmp -lm -o reduce_stddev_omp reduce_stddev_omp.c
 * Ejecutar: ./reduce_stddev_omp <num_elements>
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Crea un arreglo de num_elements números aleatorios en [0, 1] */
float *create_rand_nums(int num_elements) {
    float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
    for (int i = 0; i < num_elements; i++) {
        rand_nums[i] = (float)rand() / (float)RAND_MAX;
    }
    return rand_nums;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Uso: stddev num_elements\n");
        return 1;
    }

    int n = atoi(argv[1]);

    /* Misma semilla que la versión MPI original para reproducibilidad */
    srand(42);
    float *rand_nums = create_rand_nums(n);

    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    printf("Ejecucion con %d threads OpenMP:\n", num_threads);

    double tiempo_inicio = omp_get_wtime();

    /* --- Paso 1: calcular la media --- */
    float sum = 0.0f;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++)
        sum += rand_nums[i];

    float mean = sum / n;

    /* --- Paso 2: calcular la suma de diferencias cuadráticas --- */
    float sq_diff = 0.0f;
#pragma omp parallel for reduction(+:sq_diff)
    for (int i = 0; i < n; i++) {
        float diff = rand_nums[i] - mean;
        sq_diff += diff * diff;
    }

    double tiempo_final = omp_get_wtime();

    float stddev = sqrtf(sq_diff / n);
    printf("Mean - %f, Standard deviation = %f\n", mean, stddev);
    printf("Tiempo de ejecucion: %f segundos\n", tiempo_final - tiempo_inicio);

    free(rand_nums);
    return 0;
}
