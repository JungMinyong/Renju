#ifndef MCTS_H
#define MCTS_H

// Define constants
#define SIZE 15
#define NUM_GAMES 1000
#define BLACK 'X'
#define WHITE 'O'
//
// // Function prototypes
__global__ void mcts_kernel(char *d_board, int *d_scores);
__device__ int perform_mcts(char board[SIZE][SIZE]);
//
 #endif
//
