/*
 * reduce_avg_cuda.cu
 * Cálculo del promedio de un arreglo de números aleatorios usando CUDA.
 * Se implementa una reducción paralela en memoria compartida (shared memory)
 * con el patrón "butterfly": cada bloque produce una suma parcial, y la
 * acumulación final se realiza en CPU sobre los resultados de bloque.
 *
 * Compilar: nvcc -O2 -o reduce_avg_cuda reduce_avg_cuda.cu
 * Ejecutar: ./reduce_avg_cuda <num_elements>
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

/*
 * Reducción por bloques usando memoria compartida.
 * Cada bloque calcula la suma de su segmento y la escribe en partial_sums.
 */
__global__ void reduce_sum_kernel(const float *data, float *partial_sums, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();

    /* Reducción logarítmica dentro del bloque */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        partial_sums[blockIdx.x] = sdata[0];
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Uso: avg num_elements\n");
        return 1;
    }

    int n = atoi(argv[1]);
    srand((unsigned int)time(NULL));

    /* Generar datos en host */
    float *h_data = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
        h_data[i] = (float)rand() / RAND_MAX;

    /* Copiar datos al dispositivo */
    float *d_data;
    cudaMalloc((void **)&d_data, n * sizeof(float));
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    float *d_partial;
    cudaMalloc((void **)&d_partial, blocks * sizeof(float));

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);

    reduce_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        d_data, d_partial, n);

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);

    /* Reducción final en CPU */
    float *h_partial = (float *)malloc(blocks * sizeof(float));
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float),
               cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (int i = 0; i < blocks; i++)
        total += h_partial[i];

    printf("Total sum = %f, avg = %f\n", total, total / n);
    printf("Tiempo de ejecucion GPU: %f segundos\n", ms / 1000.0f);

    free(h_data);
    free(h_partial);
    cudaFree(d_data);
    cudaFree(d_partial);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    return 0;
}
