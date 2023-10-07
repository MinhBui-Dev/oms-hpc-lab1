#include <unistd.h>
#include "listutils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void updateRanks(long* rank_in, long* rank_out, const long* next, size_t n)
{
	/* printf("rank_in: ");
	for (size_t i = 0; i < n; i++) {
		printf("%ld ", rank_in[i]);
	}
	printf("\n"); */

	// Update the rank_out array using the rank_in array and next array using parallel for loop
	size_t i;
	#pragma omp parallel for shared(rank_in, rank_out, next, n)
	for (i = 0; i < n; i++) {
		if (next[i] != NIL)
		{
			rank_out[next[i]] = rank_in[i] + rank_in[next[i]];
		}	
	}

	/* printf("rank_out: ");
	for (size_t i = 0; i < n; i++) {
		printf("%ld ", rank_out[i]);
	}
	printf("\n"); */
}

void jumpList(long* next_in, long* next_out, size_t n)
{	
	// Update the next_out array using the next_in array using parallel for loop

	//print next_in array
	/* printf("next_in: ");
	for (size_t i = 0; i < n; i++) {
		printf("%ld ", next_in[i]);
	}
	printf("\n"); */

	size_t i;
	#pragma omp parallel for shared(next_in, next_out, n)
	for (i = 0; i < n; i++) {
		if (next_in[i] != NIL)
		{
			next_out[i] = next_in[next_in[i]];
		}
		else
		{
			next_out[i] = NIL;
		}
	}

	/* printf("next_out: ");
	for (size_t i = 0; i < n; i++) {
		printf("%ld ", next_out[i]);
	}
	printf("\n"); */

}

// swap function to swap two long pointers
void swap(long **arr1, long **arr2, size_t size)
{
    memcpy(*arr1, *arr2, size * sizeof(long));
}

void parallelListRanks (long head, const long* next, long* rank, size_t n)
{
	/* Your Code Goes Here */
	// Wyllie's Algorithm for List Ranking
	// Create two arrays of rank lists
	long* rank1 = malloc(n * sizeof(long));
	long* rank2 = malloc(n * sizeof(long));
	
	// Create two temporary arrays to store the next values
	long* next1 = malloc(n * sizeof(long));
	long* next2 = malloc(n * sizeof(long));

	// Initialize rank1 and rank2 to value 1 using parallel for loop
	// Initialize next1 and next2 to value of next using parallel for loop
	size_t i;

	#pragma omp parallel shared(next1, next2, rank1, rank2, next, n)
	{
		#pragma omp for
		for (i = 0; i < n; i++) {
		rank1[i] = 1;
		rank2[i] = 1;
		next1[i] = next[i];
		next2[i] = next[i];
		}

		#pragma omp single
		{
			// Initialize the head of the list to rank1 and rank2 to 0
			rank1[head] = 0;
			rank2[head] = 0;

			// Perform jump loop of ceiling Log2(n) iterations
			int log2n = ceil(log2(n));
			for (i = 1; i <= log2n; i++)
			{
				updateRanks(rank1, rank2, next1, n);
				jumpList(next1, next2, n);
				swap(&rank1, &rank2, n);
				swap(&next1, &next2, n);
			}

			
		}

		// Copy the final rank list to the rank array
		#pragma omp for
			for (size_t i = 0; i < n; i++) {
				rank[i] = rank1[i];
			}
	}

	free(rank1);
	free(rank2);
	free(next1);
	free(next2);
}

