#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void upSweepPhase(int N, const int two_d, const int two_d_plus_1, int* outputArray){
    int index = blockIdx.x * blockDim.x + threadIdx.x; // find thread index 
    if(index < N){
        outputArray[index*two_d_plus_1 + two_d_plus_1 - 1] += outputArray[index*two_d_plus_1 + two_d -1];
    }
}

//to allow one thread to set the last element of the array to 0.
__global__ void setLastZero(int N, int* array){
    array[N - 1] = 0;
}

__global__ void downSweepPhase(int N, const int two_d, const int two_d_plus_1, int* outputArray){
    int index = blockIdx.x * blockDim.x + threadIdx.x; // find thread index 
    if(index < N){
        int t = outputArray[index*two_d_plus_1 + two_d - 1];
        outputArray[index*two_d_plus_1 + two_d - 1] = outputArray[index*two_d_plus_1 + two_d_plus_1 - 1];
        outputArray[index*two_d_plus_1 + two_d_plus_1 - 1] += t;
    }
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int N, int* result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.

    //three main functions: upsweep phase , set to zero, downsweep phase

    int N_pow_2 = nextPow2(N);
  

    for (int two_d = 1; two_d <= N_pow_2/2; two_d*=2){
        int two_dplus1 = 2*two_d; 
        
        // compute how many parallel elements we actually have running
        int num_parallel_threads = (N_pow_2 % two_dplus1) ? N_pow_2/two_dplus1 + 1: N_pow_2/two_dplus1 ; // since we can assume that N_pow_2 is actually a power of 2

        // compute how many blocks we actually have running
        int num_blocks = (num_parallel_threads % THREADS_PER_BLOCK) ? num_parallel_threads/THREADS_PER_BLOCK + 1: num_parallel_threads/THREADS_PER_BLOCK;

        upSweepPhase<<<num_blocks, THREADS_PER_BLOCK>>>(num_parallel_threads, two_d, two_dplus1, result);

        cudaDeviceSynchronize();
    }

    setLastZero<<<1, 1>>>(N_pow_2, result);
    cudaDeviceSynchronize();

    for (int two_d = N_pow_2/2; two_d >= 1; two_d /=2){
        int two_dplus1 = 2*two_d; 
        
        // compute how many parallel elements we actually have running
        int num_parallel_threads = (N_pow_2 % two_dplus1) ? N_pow_2/two_dplus1 + 1: N_pow_2/two_dplus1 ; // since we can assume that N_pow_2 is actually a power of 2

        // compute how many blocks we actually have running
        int num_blocks = (num_parallel_threads % THREADS_PER_BLOCK) ? num_parallel_threads/THREADS_PER_BLOCK + 1: num_parallel_threads/THREADS_PER_BLOCK;

        downSweepPhase<<<num_blocks, THREADS_PER_BLOCK>>>(num_parallel_threads, two_d, two_dplus1, result);

        cudaDeviceSynchronize();
    }
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    // int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    // cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    // cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    // in place execution of exclusive scan
    exclusive_scan(N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}

__global__ void setRepeatFlags(int* input, int* output, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x; // find thread index 
    if(index < N - 1){
        output[index] = (input[index] == input[index + 1])? 1: 0; // ternary operations shorthand
    }
}

__global__ void calcNumMatches(int N, int* scans, int* flags, int* result){
    *result = scans[N - 2] + flags[N-2];
}

__global__ void gatherIndices(int N, int* scans, int* flags, int* output){
    int index = blockIdx.x * blockDim.x + threadIdx.x; // find thread index 
    if (index < N - 1){
        if(flags[index]) output[scans[index]] = index;
    }
}
// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    // count how many elements are equal to the next one, setting flags to each element that holds 
    const int length_pow_2 = nextPow2(length); //
    int* flags_array;
    int* scan_output;
    int* num_matches;

    cudaMalloc(&flags_array, length_pow_2 * sizeof(int));
    cudaMalloc(&scan_output, length_pow_2 * sizeof(int));
    cudaMalloc(&num_matches, sizeof(int));

    int num_blocks = (length_pow_2% THREADS_PER_BLOCK) ? length_pow_2/THREADS_PER_BLOCK + 1: length_pow_2/THREADS_PER_BLOCK;


    setRepeatFlags<<<num_blocks, THREADS_PER_BLOCK>>>(device_input, flags_array, length);
    
    cudaDeviceSynchronize();
    
    //copy values into second intermediate array
    cudaMemcpy(scan_output, flags_array, length*sizeof(int), cudaMemcpyDeviceToDevice);

    exclusive_scan(length, scan_output); //perform the exclusive prefix scan

    calcNumMatches<<<1, 1>>>(length_pow_2, scan_output, flags_array, num_matches);

    gatherIndices<<<num_blocks, THREADS_PER_BLOCK>>>(length_pow_2, scan_output, flags_array, device_output);

    cudaDeviceSynchronize();

    int final_length; 

    cudaMemcpy(&final_length, num_matches, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(flags_array);
    cudaFree(scan_output);
    cudaFree(num_matches);

    return final_length;    
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
