#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <assert.h>
#include <stdio.h>
#include <time.h> 
#include <windows.h>
#include <processthreadsapi.h>
#include <iostream>

#define imin(a,b) (a<b?a:b)

const int N = 5 * 1024 * 1024;
const int threadsPerBlock = 1024;
const int blocksPerGrid = imin(1024, (N + threadsPerBlock - 1) / threadsPerBlock);

double PCFreq = 0.0;
__int64 CounterStart = 0;

void StartCounter()
{
    LARGE_INTEGER li;
    if (!QueryPerformanceFrequency(&li))
        std::cout << "QueryPerformanceFrequency failed!\n";

    PCFreq = double(li.QuadPart) / 1000.0;

    QueryPerformanceCounter(&li);
    CounterStart = li.QuadPart;
}

double GetCounter()
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return double(li.QuadPart - CounterStart) / PCFreq;
}

void init(float* vector)
{
    int tid = 0;
    int a = 1;
    while (tid < N)
    {
        vector[tid] = (float)rand() / (float)(RAND_MAX / a);
        //vector[tid] = tid;
        tid += 1;
    }
}

double scalar_product_cpu(float* a, float* b)
{
    double result = 0;
    int tid = 0;
    while (tid < N)
    {
        result += a[tid] * b[tid];
        tid += 1;
    }
    return result;
}

__global__ void dot(float* a, float* b, float* c)
{
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float   temp = 0;
    while (tid < N)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // set the cache values
    cache[cacheIndex] = temp;

    // synchronize threads in this block
    __syncthreads();

    // 8
    // i = 4

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];

        __syncthreads();

        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

int main() {
    float *a, *b;
    float* partial_c;
    float result = 0;
    float* dev_a;
    float* dev_b;
    float* dev_partial_c;

    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    printf("blocksPerGrid=%d\n", blocksPerGrid);

    cudaMalloc((void**)&dev_a, N * sizeof(float));
    cudaMalloc((void**)&dev_b, N * sizeof(float));
    cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float));

    init(a);
    init(b);

    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    StartCounter();

    result = scalar_product_cpu(a, b);

    std::cout << "time on CPU =" << GetCounter() << " miliseconds" << "\n";

    printf("a=%f, b=%f, result=%f\n", a[11], b[11], result);

    cudaEvent_t start, stop;
    float gpuTime = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    dot << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_c);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("time on GPU = %f miliseconds\n", gpuTime);

    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    result = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        //printf("%f\n", partial_c[i]);
        result += partial_c[i];
    }

    printf("a=%f, b=%f, result=%f\n", a[11], b[11], result);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //Sleep(10000);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(partial_c);
    return 0;
}
