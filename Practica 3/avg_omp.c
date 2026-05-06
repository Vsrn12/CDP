
// Versión OpenMP de avg.c
// Basada en la versión MPI original de Wes Kendall (mpitutorial.com)
//
// Programa que calcula el promedio de un arreglo de elementos en paralelo
// usando OpenMP con la cláusula reduction.
//
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <assert.h>

// Crea un arreglo de números aleatorios con valores entre 0 y 1
float *create_rand_nums(int num_elements) {
    float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
    assert(rand_nums != NULL);
    int i;
    for (i = 0; i < num_elements; i++) {
        rand_nums[i] = (rand() / (float)RAND_MAX);
    }
    return rand_nums;
}

// Calcula el promedio de un arreglo usando la directiva reduction de OpenMP.
// Cada hilo acumula una copia privada de 'sum'; al finalizar el bucle,
// todas las copias se suman en la variable compartida original.
float compute_avg_omp(float *array, int num_elements) {
    float sum = 0.f;
    int i;
    #pragma omp parallel for reduction(+:sum)
    for (i = 0; i < num_elements; i++) {
        sum += array[i];
    }
    return sum / num_elements;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Uso: avg_omp num_elementos [num_hilos]\n");
        exit(1);
    }

    int num_elements = atoi(argv[1]);
    // Si no se especifica el número de hilos, se usan todos los disponibles
    int num_threads  = (argc >= 3) ? atoi(argv[2]) : omp_get_max_threads();

    omp_set_num_threads(num_threads);

    srand(time(NULL)); // Inicializar el generador de números aleatorios

    float *rand_nums = create_rand_nums(num_elements);

    double inicio = omp_get_wtime(); // Marca de tiempo de inicio

    float avg = compute_avg_omp(rand_nums, num_elements);

    double fin = omp_get_wtime(); // Marca de tiempo de fin

    printf("Hilos        : %d\n", num_threads);
    printf("Elementos    : %d\n", num_elements);
    printf("Promedio     : %f\n", avg);
    printf("Tiempo (s)   : %.6f\n", fin - inicio);

    free(rand_nums); // Liberar la memoria del arreglo
    return 0;
}
