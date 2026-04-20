#include <stdio.h>
#include <time.h>
#include <omp.h>

int main(void)
{
    int numeroHilos;
    clock_t tiempo_inicio, tiempo_final;
    long numeroIteraciones;
    double respuesta = 0.0;

    numeroHilos = omp_get_max_threads();
    omp_set_num_threads(numeroHilos);

    printf("Ejecucion con %d threads:\n", numeroHilos);
    printf("Ingresar el numero de iteraciones: ");
    scanf("%ld", &numeroIteraciones);

    tiempo_inicio = clock();

    #pragma omp parallel for reduction(+:respuesta) schedule(static)
    for (long indice = 0; indice < numeroIteraciones; indice++) {
        double termino = 4.0 / (2.0 * indice + 1.0);

        if (indice % 2 == 0) {
            respuesta += termino;
        } else {
            respuesta -= termino;
        }
    }

    tiempo_final = clock();

    printf("La respuesta es: %.8f\n", respuesta);
    printf("Tiempo de ejecucion: %f segundos\n",
           (double)(tiempo_final - tiempo_inicio) / CLOCKS_PER_SEC);

    return 0;
}