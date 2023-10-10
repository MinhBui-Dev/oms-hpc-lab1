#include <unistd.h>
#include "listutils.h"

__global__ void subListRanks(const long head, long* next, long* rank, long* sublist_ranks, long* sublist_head, long* sublist_next, long* sublist_scratch, size_t n, size_t num_sublist)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int total_threads = blockDim.x * gridDim.x;
	
	__syncthreads();

	if (tid > n)
	{
		return;
	}

	for (size_t i = tid; i < n; i+=total_threads)
	{
		rank[i] = 1;
		if (next[i] == -1)
		{
			next[i] = -(num_sublist);
		}
		
	}
	__syncthreads();
	
}

__global__ void selectSplitter(long head, long* next, long* sublist_head, long* sublist_scratch, size_t n, size_t num_sublist)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int total_threads = blockDim.x * gridDim.x;

	__syncthreads();

	for (size_t i = tid; i < num_sublist; i+=total_threads)
	{
		
		
		if (i != 0)
		{
			size_t startIndex = (i - 1) * n/(num_sublist - 1);
			size_t endIndex = i*n/(num_sublist - 1) - 1;
			// choose random pivot
			size_t pivotIndex = startIndex + (endIndex - startIndex)/2;
			if (pivotIndex == head)
			{
				pivotIndex = endIndex;
			} else if (next[pivotIndex] < 0)
			{
				pivotIndex = endIndex;
			}
			

			sublist_head[i] = pivotIndex;
			sublist_scratch[i] = next[pivotIndex];
			next[pivotIndex] = -(i);
		} else
		{
			sublist_head[i] = head;
			sublist_scratch[i] = next[head];
			next[head] = -(i);
		}
		
	}
}

__global__ void computePartialRanks(long* next, long* rank, long* sublist_head, long* sublist_scratch, long* sublist_next, long* sublist_ranks, size_t num_sublist)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int total_threads = blockDim.x * gridDim.x;

	__syncthreads();

	for (size_t i = tid ; i < num_sublist; i+=total_threads)
	{	
		if (tid > num_sublist -1)
		{
			break;
		}

		size_t pre = sublist_head[i];
		size_t current = sublist_scratch[i];

		while(next[current] >= 0)
		{
			rank[current] = rank[pre] + rank[current];
			pre = current;
			current = next[pre];
			next[pre] = -(i);
		}

		if (next[current] != -(num_sublist))
		{
			size_t k = -(next[current]);
			sublist_next[i] = k;
			sublist_ranks[k] = rank[pre];
		} else {
			rank[current] = rank[pre] + rank[current];
			next[current] = -(i);
		}

	}
	
	__syncthreads();
}

__global__ void computeRanks(long* next, long* rank, long* sublist_ranks, size_t n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int total_threads = blockDim.x * gridDim.x;

	__syncthreads();
	for (size_t i = tid; i < n; i+=total_threads)
	{
		rank[i] = rank[i] + sublist_ranks[-(next[i])] - 1;
	}

	__syncthreads();
}

extern "C" void parallelListRanks(const long head, const long* next, long* rank, const size_t n)
{
	/* Your Code Goes Here */
	long* d_next, *d_rank;
	cudaMalloc(&d_next, n * sizeof(long));
	cudaMalloc(&d_rank, n * sizeof(long));
	cudaMemcpy(d_next, next, n * sizeof(long), cudaMemcpyHostToDevice);


	size_t threadsPerBlock = 1024;
	size_t blocksPerGrid =  (n + threadsPerBlock - 1) / threadsPerBlock;
	size_t num_sublists = blocksPerGrid * log2(n);

	if (num_sublists > n)
	{
		num_sublists = n/log2(n);
	}

	long* d_sublist_next, *d_sublist_rank, *d_sublist_head, *d_sublist_scratch;
	cudaMalloc(&d_sublist_next, num_sublists * sizeof(long));
	cudaMalloc(&d_sublist_rank, num_sublists * sizeof(long));
	cudaMalloc(&d_sublist_head, num_sublists * sizeof(long));
	cudaMalloc(&d_sublist_scratch, num_sublists * sizeof(long));
	cudaMemset(d_sublist_next, -1, num_sublists * sizeof(long));
	cudaMemset(d_sublist_rank, 0, num_sublists * sizeof(long));
	cudaMemset(d_sublist_head, 0, num_sublists * sizeof(long));
	cudaMemset(d_sublist_scratch, 0, num_sublists * sizeof(long));
	
	subListRanks<<<blocksPerGrid, threadsPerBlock>>>(head, d_next, d_rank, d_sublist_rank, d_sublist_head, d_sublist_next, d_sublist_scratch, n, num_sublists);

	cudaDeviceSynchronize();

	selectSplitter<<<blocksPerGrid, threadsPerBlock>>>(head, d_next, d_sublist_head, d_sublist_scratch, n, num_sublists);

	cudaDeviceSynchronize();

	computePartialRanks<<<blocksPerGrid, threadsPerBlock>>>(d_next, d_rank, d_sublist_head, d_sublist_scratch, d_sublist_next, d_sublist_rank, num_sublists);

	long* sublist_rank = (long*)malloc(num_sublists * sizeof(long));
	long* sublist_next = (long*)malloc(num_sublists * sizeof(long));

	cudaMemcpy(sublist_rank, d_sublist_rank, num_sublists * sizeof(long), cudaMemcpyDeviceToHost);
	cudaMemcpy(sublist_next, d_sublist_next, num_sublists * sizeof(long), cudaMemcpyDeviceToHost);


	sublist_rank[0] = 0;
	size_t current = sublist_next[0];
	sublist_rank[current] = sublist_rank[0] + sublist_rank[current];
	while (sublist_next[current] > 0)
	{
		sublist_rank[sublist_next[current]] = sublist_rank[current] + sublist_rank[sublist_next[current]];
		current = sublist_next[current];
	}

	cudaMemcpy(d_sublist_rank, sublist_rank, num_sublists * sizeof(long), cudaMemcpyHostToDevice);

	computeRanks<<<blocksPerGrid, threadsPerBlock>>>(d_next, d_rank, d_sublist_rank, n);

	cudaDeviceSynchronize();
	cudaMemcpy(rank, d_rank, n * sizeof(long), cudaMemcpyDeviceToHost);
	
	cudaFree(d_next);
	cudaFree(d_rank);
	cudaFree(d_sublist_next);
	cudaFree(d_sublist_rank);
	cudaFree(d_sublist_head);
	cudaFree(d_sublist_scratch);
	free(sublist_rank);
	free(sublist_next);

}
