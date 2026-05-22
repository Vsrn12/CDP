/*
 * pi_montecarlo_cuda.cu
 * Estimación de PI por método Monte Carlo con CUDA.
 * Cada hilo GPU genera sus propios números aleatorios con cuRAND y cuenta
 * los puntos dentro del círculo unitario. Un atomicAdd acumula el conteo
 * global en el dispositivo.
 *
 * Compilar: nvcc -O2 -lcurand -o pi_montecarlo_cuda pi_montecarlo_cuda.cu
 * Ejecutar: ./pi_montecarlo_cuda <num_samples>
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS        512

/*
 * Cada hilo inicializa su propio estado cuRAND con semilla única,
 * genera samples_per_thread puntos y usa atomicAdd para el conteo global.
 */
__global__ void monte_carlo_kernel(long samples_per_thread,
                                   unsigned long long *d_count,
                                   unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    curandState state;
    curand_init(seed + (unsigned long long)idx, idx, 0, &state);

    unsigned long long local_count = 0;
    for (long i = 0; i < samples_per_thread; i++) {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if (x * x + y * y <= 1.0f)
            local_count++;
    }

    atomicAdd(d_count, local_count);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Uso: ./pi_montecarlo_cuda <num_samples>\n");
        return 1;
    }

    long samples      = atol(argv[1]);
    int  total_threads = THREADS_PER_BLOCK * NUM_BLOCKS;
    long samples_per_thread = (samples + total_threads - 1) / total_threads;

    unsigned long long *d_count;
    cudaMalloc((void **)&d_count, sizeof(unsigned long long));
    cudaMemset(d_count, 0, sizeof(unsigned long long));

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);

    monte_carlo_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        samples_per_thread, d_count, 1234ULL);

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);

    unsigned long long h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);

    long actual_samples = (long)samples_per_thread * total_threads;
    double pi = 4.0 * (double)h_count / (double)actual_samples;

    printf("Count = %llu, Samples = %ld, Estimate of pi: %.5f\n",
           h_count, actual_samples, pi);
    printf("Tiempo de ejecucion GPU: %f segundos\n", ms / 1000.0);

    cudaFree(d_count);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    return 0;
}
