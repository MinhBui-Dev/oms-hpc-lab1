#include <unistd.h>
#include "listutils.h"

__global__ void subListRanks(const long head, long* next, long* rank, long* subListRanks, long* sublistHead, long* sublistNext, long* sublistScratch, size_t n, size_t num_sublist)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int total_threads = blockDim.x * gridDim.x;
	int records_per_thread = n / total_threads;
	int extra_records = n % total_threads;
	int startRecord = tid * records_per_thread;
	int endRecord = start + records_per_thread;

	if (tid < extra_records)
	{
		startRecord += tid;
		endRecord += tid + 1;
	}
	else
	{
		startRecord += extra_records;
		endRecord += extra_records;
	}

	for (size_t i = startRecord; i < endRecord; i++)
	{
		if (next[i] == -1)
		{
			next[i] = -(num_sublist);
		}
		rank[i] = 1;
		
	}

	__syncthreads();

	int sublistPerThread = num_sublist / total_threads;
	int extra_sublist = num_sublist % total_threads;
	int startSublist = tid * sublistPerThread;
	int endSublist = startSublist + sublistPerThread;
	
	if (tid < extra_sublist)
	{
		startSublist += tid;
		endSublist += tid + 1;
	}
	else
	{
		startSublist += extra_sublist;
		endSublist += extra_sublist;
	}

	for (size_t i = startSublist; i < endSublist; i++)
	{
		if (i != 0)
		{
			size_t startIndex = (i - 1) * n/(num_sublist - 1);
			size_t endIndex = i*n/(num_sublist - 1) - 1;
			// choose random pivot
			size_t pivotIndex = startIndex + (endIndex - startIndex)/2;

			sublist_head[i] = pivotIndex;
			sublist_scratch[i] = next[pivotIndex];
			next[pivotIndex] = -(i);
		}
		{
			sublist_head[i] = head;
			sublist_scratch[i] = next[head];
			next[head] = -(i);
		}
		
	}

	__syncthreads();

	for (size_t i = startSublist; i < endSublist; i++)
	{
		size_t pre = sublist_head[i];
		size_t current = sublist_scratch[i];

		while(next[current] >= 0)
		{
			rank[current] = rank[pre] + rank[current];
			pre = current;
			current = next[pre];
			next[pre] = -(i);
		}

	}

	
	
	


}

__global__ void computeRanks(long* next, long* rank, long* sublistRanks, size_t n)
{
	/* Your Code Goes Here */
}

extern "C" void parallelListRanks(const long head, const long* next, long* rank, const size_t n)
{
	/* Your Code Goes Here */
	long* d_next, *d_rank;
	cudaMalloc(&d_next, n * sizeof(long));
	cudaMalloc(&d_rank, n * sizeof(long));
	cudaMemcpy(d_next, next, n * sizeof(long), cudaMemcpyHostToDevice);


	size_t threadsPerBlock = 64;
	size_t blocksPerGrid = 30;
	size_t subListPerThread = 32;
	size_t num_sublists = threadsPerBlock * blocksPerGrid * subListPerThread;

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
