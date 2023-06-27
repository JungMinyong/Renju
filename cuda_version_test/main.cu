#include <stdio.h>
#include "mcts.h"

#define SIZE 7
#define EMPTY '.'
#define BLACK 'X'
#define WHITE 'O'
#define THREADS_PER_BLOCK 256

int main() {

    // Allocate space for the board and scores on the host
    char h_board[SIZE][SIZE] = {0};
    int h_scores[SIZE * SIZE];

    // Initialize the board ...

    // Allocate space for the board and scores on the device
    char *d_board;
    int *d_scores;
    cudaMalloc((void **)&d_board, SIZE * SIZE * sizeof(char));
    cudaMalloc((void **)&d_scores, SIZE * SIZE * sizeof(int));

    // Copy the board to the device
    cudaMemcpy(d_board, h_board, SIZE * SIZE * sizeof(char), cudaMemcpyHostToDevice);

    // Launch the MCTS kernel
    int num_blocks = (SIZE * SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    mcts_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_board, d_scores);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the scores back to the host
    cudaMemcpy(h_scores, d_scores, SIZE * SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the scores
    for (int i = 0; i < SIZE * SIZE; i++) {
        printf("Move %d: Score %d\n", i, h_scores[i]);
    }

    // Clean up
    cudaFree(d_board);
    cudaFree(d_scores);

    return 0;
}

