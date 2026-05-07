#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* Función a integrar: f(x) = x³/3 + 4x */
float f(float x) {
    return (x * x * x / 3.0f) + 4.0f * x;
}

/*
 * Calcula la integral de f en [local_a, local_b] usando la regla del trapecio
 * con local_n subintervalos de ancho h.
 */
float trap(float local_a, float local_b, int local_n, float h) {
    float integral;
    float x;
    int i;

    integral = (f(local_a) + f(local_b)) / 2.0f;
    x = local_a;
    for (i = 1; i < local_n; i++) {
        x += h;
        integral += f(x);
    }
    integral *= h;
    return integral;
}

int main(int argc, char *argv[]) {
    int rank, size;
    float a, b, h, local_a, local_b, local_integral, total;
    int n, local_n;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Solo el proceso raíz lee los argumentos de línea de comandos */
    if (rank == 0) {
        if (argc != 4) {
            fprintf(stderr, "Uso: mpirun -np <p> ./integral_mpi <a> <b> <n>\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        a = (float)atof(argv[1]);
        b = (float)atof(argv[2]);
        n = atoi(argv[3]);
    }

    /* Difundir los parámetros a todos los procesos */
    MPI_Bcast(&a, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT,   0, MPI_COMM_WORLD);

    h       = (b - a) / (float)n;
    local_n = n / size;              /* subintervalos por proceso           */
    local_a = a + rank * local_n * h;
    local_b = local_a + local_n * h;

    local_integral = trap(local_a, local_b, local_n, h);

    /* Reducción: sumar todas las integrales parciales en el proceso raíz */
    MPI_Reduce(&local_integral, &total, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (total < 0.0f) total *= -1.0f;
        printf("Con n = %d trapezoides, la estimativa\n", n);
        printf("de la integral de %.4f hasta %.4f = %.6f\n", a, b, total);
    }

    MPI_Finalize();
    return 0;
}
