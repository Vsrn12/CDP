/*
 * pi_montecarlo_mpi.c
 * Estimación de PI por método Monte Carlo con MPI.
 * Cada proceso genera su fracción de puntos aleatorios usando erand48
 * con semilla basada en el rango del proceso, luego MPI_Reduce acumula
 * los conteos locales en el proceso raíz.
 *
 * Compilar: mpicc -O2 -o pi_montecarlo_mpi pi_montecarlo_mpi.c
 * Ejecutar: mpirun -np 4 ./pi_montecarlo_mpi <num_samples>
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int world_rank, world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 2) {
        if (world_rank == 0)
            fprintf(stderr, "Uso: mpirun -np <P> ./pi_montecarlo_mpi <num_samples>\n");
        MPI_Finalize();
        return 1;
    }

    long samples       = atol(argv[1]);
    long local_samples = samples / world_size;  /* muestras por proceso */

    /* Semilla única por proceso para erand48 */
    unsigned short xi[3] = {1, 1, (unsigned short)world_rank};

    double tiempo_inicio = MPI_Wtime();

    long local_count = 0;
    for (long i = 0; i < local_samples; i++) {
        double x = erand48(xi);
        double y = erand48(xi);
        if (x * x + y * y <= 1.0)
            local_count++;
    }

    long global_count = 0;
    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);

    double tiempo_final = MPI_Wtime();

    if (world_rank == 0) {
        long total_samples = local_samples * world_size;
        double pi = 4.0 * (double)global_count / (double)total_samples;
        printf("Count = %ld, Samples = %ld, Estimate of pi: %.5f\n",
               global_count, total_samples, pi);
        printf("Tiempo de ejecucion: %f segundos\n", tiempo_final - tiempo_inicio);
    }

    MPI_Finalize();
    return 0;
}
