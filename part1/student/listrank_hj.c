#include <unistd.h>
#include "listutils.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

void parallelListRanks (long head, const long* next, long* rank, size_t n)
{
	//Helman and Jájá’s list ranking algorithm
	//Sequentially, partition the input list into *s* sublists, by randomly choosing *s* sublist head nodes.
	//In parallel, traverse each sublist computing the list ranking of each node within the sublists.
	//Sequentially, compute the list ranking of the head nodes overall (by doing a prefix sum of the head node ranks).
	//In parallel, traverse the sublists again to convert the sublist ranking to the complete list ranking (by adding the previously determined prefix sum values).

	size_t i, num_threads, num_sublists;
	long *sublist_heads, *sublist_scratch, *sublist_next, *sublist_rank;
	long *next1 = malloc(n*sizeof(long));
	long *rank1 = malloc(n*sizeof(long));

	#pragma omp parallel shared( num_sublists, sublist_heads, head, next, next1, rank1, num_threads)
	{
		

		#pragma omp single
		{
			num_threads = omp_get_num_threads();
			num_sublists = num_threads*ceil(log2(n));
			sublist_heads = malloc((num_sublists)*sizeof(long));
			sublist_scratch = malloc((num_sublists)*sizeof(long));
			sublist_next = malloc((num_sublists)*sizeof(long));
			sublist_rank = malloc((num_sublists)*sizeof(long));
			memset(sublist_heads,0,sizeof(long)*num_sublists);
			memset(sublist_scratch,0,sizeof(long)*num_sublists);
			memset(sublist_next,-1,sizeof(long)*num_sublists);
			memset(sublist_rank,0,sizeof(long)*num_sublists);
		}

		#pragma omp for
		for (i = 0; i < n; i++)
		{
			next1[i] = next[i];
			if (next1[i] == -1)
			{
				next1[i] = -(num_sublists);
			}
			rank1[i] = 1;
		}
		
	

		size_t threadID = omp_get_thread_num();
		size_t sublistsPerThread = (num_sublists)/num_threads;
		size_t extraSublists = num_sublists % num_threads;
		size_t startSegment = threadID * sublistsPerThread;
		size_t endSegment = (threadID+1)*sublistsPerThread - 1;

		//ThreadID 0 starts with the head of the list. Account for the extra sublists by adding an additional sublist to each threadID less than extra sublist
		if (threadID < extraSublists)
		{
			startSegment += threadID;
			endSegment += threadID + 1;
		} else
		{
			startSegment += extraSublists;
			endSegment += extraSublists;
		}
		
		for (size_t j = startSegment; j <= endSegment; j++)
		{
			if( j != 0)
			{
				size_t startIndex = (j-1)*n/(num_sublists-1);
				size_t endIndex = j*n/(num_sublists-1) - 1;
				size_t randIndex = startIndex + rand()%(endIndex - startIndex + 1);
				if (randIndex == head)
				{
					randIndex = endIndex;
				} else if (next1[randIndex] < 0)
				{
					randIndex = endIndex;
				}
				
				sublist_heads[j] = randIndex;
				sublist_scratch[j] = next1[randIndex];
				next1[randIndex] = (-1)*j;
			} else
			{
				sublist_heads[j] = head;
				sublist_scratch[j] = next1[head];
				next1[head] = (-1)*j;
			}
		}
		#pragma omp barrier


		for (size_t j = startSegment; j <= endSegment; j++)
		{
			size_t pre = sublist_heads[j];
			size_t current = sublist_scratch[j];
			
			while (next1[current] >= 0)
			{
				rank1[current] = rank1[pre] + rank1[current];
				pre = current;
				current = next1[pre];
				next1[pre] = (-1)*j;
			}
			
			if (next1[current] != -(num_sublists))
			{
				size_t k = -1*next1[current];
				sublist_next[j] = k;
				sublist_rank[k] = rank1[pre];
			} else
			{
				rank1[current] = rank1[pre] + rank1[current];
				next1[current] = (-1)*j;
			}
			
		}
		#pragma omp barrier

		#pragma omp single
		{
			sublist_rank[0] = 0;
			size_t current = sublist_next[0];
			sublist_rank[current] = sublist_rank[0] + sublist_rank[current];
			while (sublist_next[current] > 0)
			{
				sublist_rank[sublist_next[current]] = sublist_rank[current] + sublist_rank[sublist_next[current]];
				current = sublist_next[current];
			}
		}
		
		size_t recordPerThread = n/num_threads;
		size_t extraRecords = n % num_threads;
		size_t startIndex = threadID*recordPerThread;
		size_t endIndex = (threadID+1)*recordPerThread - 1;

		if (threadID < extraRecords)
		{
			startIndex += threadID;
			endIndex += threadID + 1;
		} else
		{
			startIndex += extraRecords;
			endIndex += extraRecords;
		}

		for (size_t j = startIndex; j <= endIndex; j++)
		{
			rank[j] = rank1[j]+sublist_rank[(-1)*next1[j]] - 1;
		}
		#pragma omp barrier
	}
	free(sublist_heads);
	free(sublist_scratch);
	free(sublist_next);
	free(sublist_rank);
	free(next1);
	free(rank1);
}
