#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char *argv[])
{
    int lnom;
    char nombrepr[MPI_MAX_PROCESSOR_NAME];
    int pid, npr;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npr);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Get_processor_name(nombrepr, &lnom);

    printf(">> Proceso %2d de %2d activado en %s\n", pid, npr, nombrepr);

    MPI_Finalize();
    return 0;
}