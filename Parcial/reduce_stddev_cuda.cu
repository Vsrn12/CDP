/*
 * reduce_stddev_cuda.cu
 * Cálculo de la desviación estándar de un arreglo de números aleatorios
 * usando CUDA.
 * Paso 1: kernel de reducción para la suma (→ media).
 * Paso 2: kernel de reducción de diferencias cuadráticas (→ stddev).
 * Ambos kernels usan reducción logarítmica en shared memory.
 *
 * Compilar: nvcc -O2 -o reduce_stddev_cuda reduce_stddev_cuda.cu
 * Ejecutar: ./reduce_stddev_cuda <num_elements>
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/* Reducción de suma por bloques */
__global__ void reduce_sum_kernel(const float *data, float *partial_sums, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        partial_sums[blockIdx.x] = sdata[0];
}

/* Reducción de (data[i] - mean)^2 por bloques */
__global__ void reduce_sq_diff_kernel(const float *data, float mean,
                                      float *partial_sums, int n) {
    extern __shared__ float sdata[];

    int tid  = threadIdx.x;
    int idx  = blockIdx.x * blockDim.x + tid;
    float d  = (idx < n) ? (data[idx] - mean) : 0.0f;

    sdata[tid] = d * d;
    __syncthreads();

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
        fprintf(stderr, "Uso: stddev num_elements\n");
        return 1;
    }

    int n = atoi(argv[1]);
    srand(42);

    float *h_data = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
        h_data[i] = (float)rand() / RAND_MAX;

    float *d_data;
    cudaMalloc((void **)&d_data, n * sizeof(float));
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    float *d_partial;
    cudaMalloc((void **)&d_partial, blocks * sizeof(float));
    float *h_partial = (float *)malloc(blocks * sizeof(float));

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);

    /* --- Paso 1: suma para calcular la media --- */
    reduce_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        d_data, d_partial, n);
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float),
               cudaMemcpyDeviceToHost);

    float total_sum = 0.0f;
    for (int i = 0; i < blocks; i++)
        total_sum += h_partial[i];
    float mean = total_sum / n;

    /* --- Paso 2: suma de diferencias cuadráticas --- */
    reduce_sq_diff_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        d_data, mean, d_partial, n);
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float),
               cudaMemcpyDeviceToHost);

    float sq_diff = 0.0f;
    for (int i = 0; i < blocks; i++)
        sq_diff += h_partial[i];

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);

    float stddev = sqrtf(sq_diff / n);
    printf("Mean - %f, Standard deviation = %f\n", mean, stddev);
    printf("Tiempo de ejecucion GPU: %f segundos\n", ms / 1000.0f);

    free(h_data);
    free(h_partial);
    cudaFree(d_data);
    cudaFree(d_partial);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    return 0;
}
