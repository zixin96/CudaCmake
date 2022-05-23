#include "common/book.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024;
// How many threads per block should we launch?
// A fixed value because the number of threads per block is limited
const int threadsPerBlock = 256;

// How many blocks should we launch? There are multiple things to consider 
// Firstly, in the final step, we use CPU to sum every entries in an intermediate array of length == blocksPerGrid. Thus blocksPerGrid should be manageably small for the CPU.
// Secondly, blocksPerGrid should be large enough to keep GPU busy
// Here, we choose a fixed value of 32 blocks. You may notice better/worse performance for other choices.
// What the hack is this imin thing?
// If the size of the given input vector N is too small, 32 blocks of 256 threads apiece is too much.
// In this case, we need the smallest multiple of 256 >= N.
// Finally, we choose the smaller between 32 and this smallest multiple to ensure we don't spawn too many blocks
const int blocksPerGrid =
imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);


// a: dot product input vector
// b: dot product input vector
// c: an array storing intermediate results of a dot product where each entry contains the sum produced by one of the parallel blocks
__global__ void dot(float* a, float* b, float* c)
{
	// we declare a buffer of shared memory named cache.

	// Each block has its own private copy of this shared memory

	// Every thread in that block shares the memory, but threads cannot see or modify the copy of this variable that is seen within other blocks

	// This provides a great way for threads to communicate with each other. Moreover, shared memory buffers reside physically on the GPU,
	// the latency to access shared memory tends to be far lower than typical buffers. Thus, shared memory is effectively a per-block,
	// software-managed cache or scratchpad

	// Because computing dot product needs intermediate the result to be the sum of all these pairwise products,
	// each thread keeps a running sum (cache) of the pairs it has added

	// every thread in the block keeps a running sum, thus the size of the cache matches the number of threads per block
	__shared__ float cache[threadsPerBlock];
	// Given our configuration (similar to Figure 5.1, Page 64), compute our data indices
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// the offset in our shared memory cache is just our thread index
	int cacheIndex = threadIdx.x;

	// temp is used to store each thread's running sum
	float temp = 0;
	while (tid < N)
	{
		temp += a[tid] * b[tid];
		// each thread increments their indices by the total number of threads to ensure we don't miss any elements and don't multiply a pair twice
		tid += blockDim.x * gridDim.x;
	}

	// set the cache values
	cache[cacheIndex] = temp;

	// If we expect to communicate between threads, we also need a mechanism for synchronizing between threads
	// For example, if thread A writes a value to shared memory and we want thread B to do something with this value,
	// we can’t have thread B start its work until we know the write from thread A is complete.
	// Without synchronization, we have created a race condition where the correctness of the execution results depends on the non-deterministic details of the hardware

	// synchronize threads in this block
	// this call guarantees that every thread in the block has completed instructions prior to the __syncthreads() before the hardware will execute the next instruction on any thread
	__syncthreads();
	// At this point, we know that our temporary cache has been filled, we can sum the values in it. 

	// Next, we need to sum the values in the cache.
	// We could have one thread iterate over the cache and calculate a running sum, but this takes O(n) where n is the size of the cache
	// Since we have hundreds of threads available to do our work, we can do this in parallel in O(log(n)) time:
	// each thread adds two of the values in cache[] and store the result back to cache[]. Since each thread combines 2 entries into 1,
	// we complete this step with half as many entries as we started with. In the next step, we do the same thing on the remaining half.
	// We continue this fashion for log_2(cacheSize) steps until we have the sum of every entry in cache[]. See Markdown. 

	// threadsPerBlock (cacheSize) must be a power of 2

	// For the first step, we start with i as half the number of threads per block (for cache size of 8, i = 4)
	int i = blockDim.x / 2;
	// we continue the process until we have accumulated the value in cache[0]
	while (i != 0)
	{
		// we only want the threads with indices < i to do any work (for the first step, that's i = 0,1,2,3)
		if (cacheIndex < i)
		{
			// following the summation reduction pattern (Figure 5.4 Page 80),
			// each thread will compute the sum of the entry at (cacheIndex) and (cacheIndex + i) and store that sum back to cache[cacheIndex]
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		// before we process to the next step, we need to ensure that every threads that needs to write to cache[] has finished doing so
		__syncthreads();
		i /= 2;
	}

	// PS: this process is called reduction, because it involves taking an input array and produce a smaller array (a single value)

	// after termination of the while loop, each block has a single summation value containing every pairwise product the threads in that block computed
	// Why do we do this global store only for the thread with cacheIndex == 0?
	// Since there is only one number that needs writing to global memory, only a single thread needs to perform this operation
	// (you can of course choose any cacheIndex to write cache[0] to global memory)
	if (cacheIndex == 0)
	{
		// store this single value to global memory array where each element is a partial summation of the whole dot product
		c[blockIdx.x] = cache[0];
	}

	// Since c only contains the intermediate results of a dot product (we need to sum every entries in c to get the final result),
	// why do we exit the kernel without performing this last step (and return control to the host and let CPU finish the job)?
	// Because GPU tends to waste its resources while performing reductions on small data set (in this case, c only contains 32 floats),
	// we delegate this final step to CPU so that GPU is free to start another dot product or work on another large computation
}


int main(void)
{
	float* a, * b, c, * partial_c;
	float* dev_a, * dev_b, * dev_partial_c;

	// allocate memory on the cpu side
	// CPU-side dot product input vector
	a = (float*)malloc(N * sizeof(float));
	// CPU-side dot product input vector
	b = (float*)malloc(N * sizeof(float));
	// CPU-side intermediate results for dot product 
	partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

	// similar vectors are allocated in GPU-side
	HANDLE_ERROR(cudaMalloc((void**)&dev_a,
		N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b,
		N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c,
		blocksPerGrid * sizeof(float)));

	// fill in the host memory with data
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * 2;
	}

	// copy the arrays 'a' and 'b' to the GPU
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float),
		cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float),
		cudaMemcpyHostToDevice));

	dot << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b,
		dev_partial_c);

	// copy the intermediate results from the GPU to the CPU
	HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c,
		blocksPerGrid * sizeof(float),
		cudaMemcpyDeviceToHost));

	// the last step of the dot product is to sum the entries in CPU-side partial_c
	c = 0;
	for (int i = 0; i < blocksPerGrid; i++)
	{
		c += partial_c[i];
	}

#define sum_squares(x)  (x*(x+1)*(2*x+1)/6)
	printf("Does GPU value %.6g = %.6g?\n", c,
		2 * sum_squares((float)(N - 1)));

	// free memory on the gpu side
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_partial_c));

	// free memory on the cpu side
	free(a);
	free(b);
	free(partial_c);
}
