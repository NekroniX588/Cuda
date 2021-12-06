
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
    case 2: // Fermi
        if (devProp.minor == 1) cores = mp * 48;
        else cores = mp * 32;
        break;
    case 3: // Kepler
        cores = mp * 192;
        break;
    case 5: // Maxwell
        cores = mp * 128;
        break;
    case 6: // Pascal
        if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
        else if (devProp.minor == 0) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    case 7: // Volta and Turing
        if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    case 8: // Ampere
        if (devProp.minor == 0) cores = mp * 64;
        else if (devProp.minor == 6) cores = mp * 128;
        else printf("Unknown device type\n");
        break;
    default:
        printf("Unknown device type\n");
        break;
    }
    return cores;
}



void print_prop(cudaDeviceProp &prop) {
    printf("Name: %s \n", prop.name);
    printf("Global memory available on device in bytes: %zu \n", prop.totalGlobalMem);
    printf("Constant memory available on device in bytes: %zu \n", prop.totalConstMem);
    printf("Shared memory available per block in bytes: %zu \n", prop.sharedMemPerBlock);
    printf("multiProcessorCount: %d \n", prop.multiProcessorCount);
    printf("Maximum size of each dimension of a grid[0], %d \n", prop.maxGridSize[0]);
    printf("Maximum size of each dimension of a grid[1], %d \n", prop.maxGridSize[1]);
    printf("Maximum size of each dimension of a grid[2], %d \n", prop.maxGridSize[2]);
    printf("Maximum number of threads per block, %d \n", prop.maxThreadsPerBlock);
    printf("Maximum size of each dimension of a block[0], %d \n", prop.maxThreadsDim[0]);
    printf("Maximum size of each dimension of a block[1], %d \n", prop.maxThreadsDim[1]);
    printf("Maximum size of each dimension of a block[2], %d \n", prop.maxThreadsDim[2]);
    printf("Count of cores: %d \n", getSPcores(prop));
}

int main()
{
    int count;
    cudaDeviceProp prop;

    cudaGetDeviceCount(&count);

    printf("GPUS count: %d \n", count);

    cudaGetDeviceProperties(&prop, 0);

    print_prop(prop);


    return 0;
}