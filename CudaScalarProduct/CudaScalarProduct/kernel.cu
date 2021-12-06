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

const int N = 10*1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

namespace gpu_sp {
    __global__ void init(float* a) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        while (tid < N) {
            a[tid] = tid;
            //printf("%f\n", a[tid]);
            tid += blockDim.x * gridDim.x;
        }
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

    void print_cuda(float *arr, float *arr_cuda, int N){
        cudaMemcpy(arr, arr_cuda, N * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 10; i++) {
            printf("%f\n", arr[i]);
        }

    }

    void gpu_main(int blocks, int treads) {
        float *a;
        float *b;
        float *partial_c;
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

        //init << <blocks, treads >> > (dev_a);
        //init << <blocks, treads >> > (dev_b);

        for (int i = 0; i < N; i++) {
            a[i] = i;
            b[i] = i * 2;
        }

        cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

       // print_cuda(a, dev_a, N);
        //print_cuda(b, dev_b, N);

        cudaEvent_t start, stop;
        float gpuTime = 0.0;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        dot <<<blocksPerGrid, threadsPerBlock >>> (dev_a, dev_b, dev_partial_c);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        printf("time on GPU = %f miliseconds\n", gpuTime);

        cudaMemcpy(a, dev_a, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(b, dev_b, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

        printf("a=%f, b=%f, result=%f\n", a[11], b[11], result);

        

        for (int i = 0; i < blocksPerGrid; i++) {
            //printf("%f\n", partial_c[i]);
            result += partial_c[i];
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("a=%f, b=%f, result=%f\n", a[11], b[11], result);
        #define sum_squares(x)  (x*(x+1)*(2*x+1)/6)
                printf("Does GPU value %f = %f?\n", result,
                    2 * sum_squares((float)(N - 1)));
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(partial_c);
    }
}

namespace cpu_sp {
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
        int a = 10;
        while (tid < N)
        {
            //vector[tid] = (float)rand() / (float)(RAND_MAX / a);
            vector[tid] = tid;
            tid += 1;  
        }
    }

    double scalar_product(float* a, float* b)
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

    void cpu_main(void)
    {
        float a[N], b[N];
        double c;

        // fill the arrays 'a' and 'b' on the CPU
        //init(a);
        //init(b);
        for (int i = 0; i < N; i++) {
            a[i] = i;
            b[i] = i * 2;
        }

        StartCounter();

        c = scalar_product(a, b);

        std::cout << "CPU time" << GetCounter() << "\n";

        printf("a=%f, b=%f, c=%f\n", a[11], b[11], c);

    }
}

int main(){
    cpu_sp::cpu_main();
    gpu_sp::gpu_main(16,1024);
	return 0;
}