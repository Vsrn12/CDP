/*
 * reduce_avg_omp.c
 * Cálculo del promedio de un arreglo de números aleatorios usando OpenMP.
 * Se utiliza la cláusula reduction(+:global_sum) para acumular la suma
 * parcial de cada hilo en una sola variable de forma segura.
 *
 * Compilar: gcc -O2 -fopenmp -o reduce_avg_omp reduce_avg_omp.c
 * Ejecutar: ./reduce_avg_omp <num_elements>
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

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
        fprintf(stderr, "Uso: avg num_elements\n");
        return 1;
    }

    int num_elements = atoi(argv[1]);

    srand((unsigned int)time(NULL));
    float *rand_nums = create_rand_nums(num_elements);

    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    printf("Ejecucion con %d threads OpenMP:\n", num_threads);

    double tiempo_inicio = omp_get_wtime();

    /* Reducción paralela: cada hilo acumula su suma parcial, luego
       OpenMP las combina automáticamente */
    float global_sum = 0.0f;
#pragma omp parallel for reduction(+:global_sum)
    for (int i = 0; i < num_elements; i++) {
        global_sum += rand_nums[i];
    }

    double tiempo_final = omp_get_wtime();

    printf("Total sum = %f, avg = %f\n",
           global_sum, global_sum / num_elements);
    printf("Tiempo de ejecucion: %f segundos\n", tiempo_final - tiempo_inicio);

    free(rand_nums);
    return 0;
}
