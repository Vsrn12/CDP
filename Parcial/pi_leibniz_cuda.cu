/*
 * pi_leibniz_cuda.cu
 * Cálculo de PI usando la serie de Leibniz con CUDA.
 * Cada hilo GPU calcula los términos que le corresponden (distribución
 * por stride), y el resultado parcial se acumula en CPU.
 *
 * Compilar: nvcc -O2 -o pi_leibniz_cuda pi_leibniz_cuda.cu
 * Ejecutar: ./pi_leibniz_cuda
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS        1024

/*
 * Cada hilo recorre los índices con paso = total_threads.
 * El resultado parcial de cada hilo se almacena en partial_sums[idx].
 */
__global__ void leibniz_kernel(long n, double *partial_sums) {
    long idx    = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x  * blockDim.x;

    double local_sum = 0.0;
    for (long i = idx; i < n; i += stride) {
        if (i % 2 == 0) {
            local_sum += 4.0 / (2.0 * i + 1.0);
        } else {
            local_sum -= 4.0 / (2.0 * i + 1.0);
        }
    }
    partial_sums[idx] = local_sum;
}

int main(void) {
    long n;
    printf("Ingresar el numero de iteraciones: ");
    fflush(stdout);
    scanf("%ld", &n);

    long total_threads = (long)THREADS_PER_BLOCK * NUM_BLOCKS;

    /* Memoria en dispositivo para sumas parciales */
    double *d_partial;
    cudaMalloc((void **)&d_partial, total_threads * sizeof(double));
    cudaMemset(d_partial, 0, total_threads * sizeof(double));

    /* Medición de tiempo GPU */
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);

    leibniz_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(n, d_partial);

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);

    /* Copiar sumas parciales a host y acumular */
    double *h_partial = (double *)malloc(total_threads * sizeof(double));
    cudaMemcpy(h_partial, d_partial, total_threads * sizeof(double),
               cudaMemcpyDeviceToHost);

    double respuesta = 0.0;
    for (long i = 0; i < total_threads; i++) {
        respuesta += h_partial[i];
    }

    printf("La respuesta es: %.8f\n", respuesta);
    printf("Tiempo de ejecucion GPU: %f segundos\n", ms / 1000.0);

    free(h_partial);
    cudaFree(d_partial);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    return 0;
}
