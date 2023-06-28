#include "mcts.h"

__global__ void mcts_kernel(char *d_board, int *d_scores) {
    // Each thread represents a different initial move
    int move = threadIdx.x + blockIdx.x * blockDim.x;

    // Copy the current board to local memory
    char board[SIZE][SIZE];
    for (int i = 0; i < SIZE * SIZE; i++) {
        board[i / SIZE][i % SIZE] = d_board[i];
    }

    // Make the initial move
    board[move / SIZE][move % SIZE] = BLACK;

    // Perform MCTS from the resulting state
    int score = perform_mcts(board);

    // Write the score back to global memory
    d_scores[move] = score;
}

__device__ int perform_mcts(char board[SIZE][SIZE]) {
    // Initialize the score
    int score = 0;

    // Perform a large number of random games
    for (int i = 0; i < NUM_GAMES; i++) {
        // Copy the board to local memory
        char local_board[SIZE][SIZE];
        for (int j = 0; j < SIZE * SIZE; j++) {
            local_board[j / SIZE][j % SIZE] = board[j / SIZE][j % SIZE];
        }

        // Play a random game to the end
        score += 10; //play_random_game(local_board);
    }

    // Return the average score
    return score / NUM_GAMES;
}

