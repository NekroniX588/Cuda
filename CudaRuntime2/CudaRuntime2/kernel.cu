
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <time.h> 
#include <stdio.h>
#include <windows.h>
#include <processthreadsapi.h>
#include <iostream>


#define N 65000

namespace gpu_add
{
    __global__ void init(int* a) {
        int tid = blockIdx.x;
        if (tid < N) {
            a[tid] = tid;
        }
    }

    __global__ void add_block(int* a, int* b, int* c) {
        int tid = blockIdx.x;
        if (tid < N) {
            c[tid] = a[tid] + b[tid];
        }
    }

    __global__ void add(int* a, int* b, int* c) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        while (tid < N) {
            c[tid] = a[tid] + b[tid];
            tid += gridDim.x * blockDim.x;
        }
    }

    void gpu_main (int blocks, int treads){
        int a[N];
        int b[N];
        int c[N];
        int* dev_a;
        int* dev_b;
        int* dev_c;

        printf("blocks=%d, treads=%d\n", blocks, treads);

        cudaMalloc((void**)&dev_a, N * sizeof(int));
        cudaMalloc((void**)&dev_b, N * sizeof(int));
        cudaMalloc((void**)&dev_c, N * sizeof(int));


        init << <N, 1 >> > (dev_a);
        init << <N, 1 >> > (dev_b);
        cudaEvent_t start, stop;
        float gpuTime = 0.0;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        //add_block << <N, 1>> > (dev_a, dev_b, dev_c);
        add << <blocks, treads >> > (dev_a, dev_b, dev_c);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        printf("time on GPU = %f miliseconds\n", gpuTime);

        cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaMemcpy(a, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(b, dev_b, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

        //printf("a=%d, b=%d, c=%d\n", a[11], b[11], c[11]);

        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
    }
}


namespace cpu_add
{
    double PCFreq = 0.0;
    __int64 CounterStart = 0;

    double get_cpu_time() {
        FILETIME a, b, c, d;
        if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0) {
            //  Returns total user time.
            //  Can be tweaked to include kernel times as well.
            return
                (double)(d.dwLowDateTime |
                    ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
        }
        else {
            //  Handle error
            return 0;
        }
    }

    void add(int* a, int* b, int* c)
    {
        int tid = 0;    // this is CPU zero, so we start at zero
        while (tid < N)
        {
            c[tid] = a[tid] + b[tid];
            tid += 1;   // we have one CPU, so we increment by one
        }
    }

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

    void cpu_main(void)
    {
        int a[N], b[N], c[N];

        // fill the arrays 'a' and 'b' on the CPU
        for (int i = 0; i < N; i++)
        {
            a[i] = i;
            b[i] = i;
        }

        StartCounter();

        add(a, b, c);

        std::cout <<"CPU time" << GetCounter() << "\n";

        printf("a=%d, b=%d, c=%d\n", a[11],b[11],c[11]);

    }
}

int main()
{

    cpu_add::cpu_main();

    gpu_add::gpu_main(128,128);
    gpu_add::gpu_main(128, 1024);
    gpu_add::gpu_main(1024, 128);
    gpu_add::gpu_main(1024, 1024);
    return 0;
}