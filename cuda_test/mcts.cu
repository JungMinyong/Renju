#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "winfunc.h"
#include "board.h"
#include "mcts.h"

#define BLOCK_SIZE 256
#define BOARD_SIZE 7


__device__ bool is_valid_move_cu(GameState* state, int row, int col) {
    bool status = state->board[row][col] == EMPTY;
    return status;
}
__device__ ValidMoves get_valid_moves_cu(GameState* state) {
    ValidMoves valid_moves = {0};
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (is_valid_move_cu(state, i, j)) {
                valid_moves.rows[valid_moves.count] = i;
                valid_moves.cols[valid_moves.count] = j;
                valid_moves.count++;
            }
        }
    }
    return valid_moves;
}
__device__ void add_stone_cu(GameState* state, char column, int row) {
    int x = column - 'a';
    int y = row - 1;

    if (x >= 0 && x < SIZE && y >= 0 && y < SIZE && state->board[y][x] == EMPTY) {
        state->board[y][x] = state->current_stone;
    }
}
__device__ bool checkWin_cu(GameState *state) {
    int player = state->current_stone;
	// Check rows
    for (int row = 0; row < BOARD_SIZE; row++) {
        for (int col = 0; col <= BOARD_SIZE - 5; col++) {
            int count = 0;
            for (int i = 0; i < 5; i++) {
                if (state->board[row][col + i] == player)
                    count++;
                else
                    break;
            }
            if (count == 5)
                return true;
        }
    }

    // Check columns
    for (int col = 0; col < BOARD_SIZE; col++) {
        for (int row = 0; row <= BOARD_SIZE - 5; row++) {
            int count = 0;
            for (int i = 0; i < 5; i++) {
                if (state->board[row + i][col] == player)
                    count++;
                else
                    break;
            }
            if (count == 5)
                return true;
        }
    }

    // Check diagonals (top-left to bottom-right)
    for (int row = 0; row <= BOARD_SIZE - 5; row++) {
        for (int col = 0; col <= BOARD_SIZE - 5; col++) {
            int count = 0;
            for (int i = 0; i < 5; i++) {
                if (state->board[row + i][col + i] == player)
                    count++;
                else
                    break;
            }
            if (count == 5)
                return true;
        }
    }

    // Check diagonals (top-right to bottom-left)
    for (int row = 0; row <= BOARD_SIZE - 5; row++) {
        for (int col = BOARD_SIZE - 1; col >= 4; col--) {
            int count = 0;
            for (int i = 0; i < 5; i++) {
                if (state->board[row + i][col - i] == player)
                    count++;
                else
                    break;
            }
            if (count == 5)
                return true;
        }
    }

    return false;
}

__device__ bool checkDraw_cu(GameState *state){
    for (int row = 0; row < BOARD_SIZE; row++) {
        for (int col = 0; col < BOARD_SIZE; col++) {
            if (state->board[row][col] == 0){
                return false;
            }       
        }
    }
    return true;
}

 
__device__ int get_winner_cu(GameState* state) {
    return checkWin_cu(state) ? state->current_stone : -1;
}

__device__ int is_game_over_cu(GameState* state) {
    return checkWin_cu(state) || checkDraw_cu(state);
}


__device__ int get_valid_move_count_cu(GameState* state) {
    return get_valid_moves_cu(state).count;
}


__device__ GameState make_move_cu(GameState* state, int action) {
    ValidMoves valid_moves = get_valid_moves_cu(state);
    int row = valid_moves.rows[action];
    int col = valid_moves.cols[action];
    add_stone_cu(state, col + 'a', row + 1);
    state->current_stone = (state->current_stone == BLACK) ? WHITE : BLACK;
    return *state;
}

__device__ GameState make_move_copy_cu(GameState state, int action) {
    ValidMoves valid_moves = get_valid_moves_cu(&state);
    int row = valid_moves.rows[action];
    int col = valid_moves.cols[action];
    add_stone_cu(&state, col + 'a', row + 1);
    state.current_stone = (state.current_stone == BLACK) ? WHITE : BLACK;
    return state;
}




__device__ int simulate_game(GameState state,int num) {
    GameState copy_state = state; // making a copy to not alter the original state

    while (!is_game_over_cu(&copy_state)) {
        int num_actions = get_valid_move_count_cu(&copy_state);
        if (num_actions == 0) {
            break;
        }
        int action = num % num_actions;
        copy_state = make_move_cu(&copy_state, action);
    }

    return get_winner_cu(&copy_state);
}

__global__ void simulate_game_cuda(GameState* states, int* winners, int num) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;


    GameState state = states[index];
    winners[index] = simulate_game(state, num);

}



int monte_carlo_tree_search(GameState original_state, int MAX_SEARCH, int num) {

    GameState state = original_state;
    int num_actions = get_valid_move_count(&state);
    int current_stone = state.current_stone;

    // Allocate memory on the host and device
    int* wins = (int*)malloc(num_actions * sizeof(int));
    int visits[num_actions];
    int* d_wins;
    cudaMalloc((void**)&d_wins, num_actions * sizeof(int));

    for (int i = 0; i < num_actions; i++) {
        wins[i] = 0;
        visits[i] = 0;
    }

    int num_blocks = (num_actions + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 gridDim(num_blocks);
    dim3 blockDim(BLOCK_SIZE);

    for (int i = 0; i < MAX_SEARCH; i++) {
        int action = rand() % num_actions;
        GameState new_state = make_move_copy(state, action);

        // Copy data to device
        cudaMemcpy(d_wins, wins, num_actions * sizeof(int), cudaMemcpyHostToDevice);

        // Launch CUDA kernel for parallel simulation

        simulate_game_cuda<<<gridDim, blockDim>>>(&new_state, &d_wins[action], num);

        // Copy data back to host
        cudaMemcpy(d_wins, wins, num_actions * sizeof(int), cudaMemcpyDeviceToHost);

        visits[action]++;
        if (current_stone == wins[action]) {
            wins[action]++;
        }
    }


    int best_action = 0;
    double best_ratio = -1;
    for (int i = 0; i < num_actions; i++) {
        double ratio = (double)wins[i] / visits[i];
        if (ratio > best_ratio) {
            best_ratio = ratio;
            best_action = i;
        }
    }


    // Free allocated memory on host and device
    free(wins);
    cudaFree(d_wins);

    return best_action;
}
