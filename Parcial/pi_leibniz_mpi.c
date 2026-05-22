/*
 * pi_leibniz_mpi.c
 * Cálculo de PI usando la serie de Leibniz con MPI.
 * Cada proceso calcula una suma parcial de los términos que le corresponden
 * (distribución cíclica), y luego MPI_Reduce acumula el resultado en el
 * proceso raíz.
 *
 * Compilar: mpicc -O2 -o pi_leibniz_mpi pi_leibniz_mpi.c
 * Ejecutar: mpirun -np 4 ./pi_leibniz_mpi
 */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int world_rank, world_size;
    long numeroIteraciones;
    double sumaLocal = 0.0, respuesta = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* Solo el proceso 0 lee la entrada y la difunde a los demás */
    if (world_rank == 0) {
        printf("Ejecucion con %d procesos MPI:\n", world_size);
        printf("Ingresar el numero de iteraciones: ");
        fflush(stdout);
        scanf("%ld", &numeroIteraciones);
    }

    MPI_Bcast(&numeroIteraciones, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    double tiempo_inicio = MPI_Wtime();

    /* Distribución cíclica: proceso p toma los índices p, p+P, p+2P, ... */
    for (long indice = world_rank; indice < numeroIteraciones; indice += world_size) {
        if (indice % 2 == 0) {
            sumaLocal += 4.0 / (2.0 * indice + 1.0);
        } else {
            sumaLocal -= 4.0 / (2.0 * indice + 1.0);
        }
    }

    /* Reducción global: suma todas las sumas parciales en el proceso 0 */
    MPI_Reduce(&sumaLocal, &respuesta, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double tiempo_final = MPI_Wtime();

    if (world_rank == 0) {
        printf("La respuesta es: %.8f\n", respuesta);
        printf("Tiempo de ejecucion: %f segundos\n", tiempo_final - tiempo_inicio);
    }

    MPI_Finalize();
    return 0;
}
