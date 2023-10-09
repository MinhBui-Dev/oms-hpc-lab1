#include <unistd.h>
#include "listutils.h"

__global__ void subListRanks(const long head, long* next, long* rank, long* sublist_ranks, long* sublist_head, long* sublist_next, long* sublist_scratch, size_t n, size_t num_sublist)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int total_threads = blockDim.x * gridDim.x;
	int records_per_thread = n / total_threads;
	int extra_records = n % total_threads;
	int startRecord = tid * records_per_thread;
	int endRecord = startRecord + records_per_thread;
	

	

	if (tid > n)
	{
		return;
	}

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
	printf("tid: %d, startRecord: %d, endRecord: %d\n", tid, startRecord, endRecord);

	__syncthreads();

	for (size_t i = startRecord; i < endRecord; i++)
	{
		printf("tid: %d, i: %ld, next[i]: %ld\n", tid, i, next[i]);
		if (next[i] == -1)
		{
			printf("tid: %d, i: %ld, next[i]: %ld\n", tid, i, next[i]);
			printf("num_sublist: %ld\n", num_sublist);
			next[i] = -(num_sublist);
		}
		rank[i] = 1;
		printf("tid: %d, i: %ld, next[i]: %ld\n", tid, i, next[i]);
		
	}
	printf("Out of loop\n");
	printf("tid: %d, head: %ld\n", tid, head);
	printf("tid: %d, next[head]: %ld\n", tid, next[head]);
	__syncthreads();

	if	(tid == 0)
	{
		printf("next: ");
		for (size_t i = 0; i < n; i++)
		{
			printf("%ld ", next[i]);
		}
		printf("\n");

	}
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
			//printf("startIndex: %ld, endIndex: %ld\n", startIndex, endIndex);
			// choose random pivot
			size_t pivotIndex = startIndex;
			if (pivotIndex == head)
			{
				pivotIndex = endIndex;
			} else if (next[pivotIndex] < 0)
			{
				/* printf("pivotIndex: %ld\n", pivotIndex);
				printf("next[pivotIndex]: %ld\n", next[pivotIndex]);
				printf("head: %ld\n", head);
				printf("startIndex: %ld, endIndex: %ld\n", startIndex, endIndex); */
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

	/* if	(tid == 0)
	{
		printf("sublist_head: ");
		for (size_t i = 0; i < num_sublist; i++)
		{
			printf("%ld ", sublist_head[i]);
		}
		printf("\n");
		printf("sublist_scratch: ");
		for (size_t i = 0; i < num_sublist; i++)
		{
			printf("%ld ", sublist_scratch[i]);
		}
		printf("\n");
		printf("next: ");
		for (size_t i = 0; i < n; i++)
		{
			printf("%ld ", next[i]);
		}
		printf("\n");

	} */

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
	int records_per_thread = n / total_threads;
	int extra_records = n % total_threads;
	int startRecord = tid * records_per_thread;
	int endRecord = startRecord + records_per_thread;

	if (tid > n)
	{
		return;
	}

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

	//size_t sublistPerThread = 4;

	size_t threadsPerBlock = 2;
	size_t blocksPerGrid =  2;
	size_t total_threads = threadsPerBlock * blocksPerGrid;
	size_t num_sublists = total_threads * log2(n);

	//size_t threadsPerBlock = 64;
	//size_t blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	//size_t subListPerThread = 32;
	//size_t total_threads = threadsPerBlock * blocksPerGrid;
	//size_t num_sublists = total_threads * log2(n);
	//size_t num_sublists = threadsPerBlock * blocksPerGrid * subListPerThread;

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

	long* sublist_rank = (long*)malloc(num_sublists * sizeof(long));
	long* sublist_next = (long*)malloc(num_sublists * sizeof(long));

	cudaMemcpy(sublist_rank, d_sublist_rank, num_sublists * sizeof(long), cudaMemcpyDeviceToHost);
	cudaMemcpy(sublist_next, d_sublist_next, num_sublists * sizeof(long), cudaMemcpyDeviceToHost);

	
	/* printf("sublist_rank: ");
	for (size_t i = 0; i < num_sublists; i++)
	{
		printf("%ld ", sublist_rank[i]);
	}
	printf("\n"); */

	/* printf("sublist_next: ");
	for (size_t i = 0; i < num_sublists; i++)
	{
		printf("%ld ", sublist_next[i]);
	}
	printf("\n");
 */
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
