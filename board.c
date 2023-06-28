#include <stdio.h>
#include "board.h"

int board[SIZE][SIZE];
int current_stone = 1; //BLACK;

void initialize_board(GameState *state) {
    for(int i = 0; i < SIZE; i++) {
        for(int j = 0; j < SIZE; j++) {
            state->board[i][j] = EMPTY;
        }
    }
    state->current_stone = BLACK;
    // initialize other fields of GameState...
}


//print current board status
bool is_valid_move(GameState *state, int row, int col){
    bool status = state->board[row][col] == EMPTY;
    return status;
}

ValidMoves get_valid_moves(GameState *state){
    ValidMoves valid_moves = {0};
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++){
            if (is_valid_move(state, i, j)){
                valid_moves.rows[valid_moves.count] = i;
                valid_moves.cols[valid_moves.count] = j;
                valid_moves.count++;
            }
        }
    }
    return valid_moves;
}

void print_board(GameState *state) {
    printf("   ");
    for(int i = 0; i < SIZE; i++) {
        printf("%c ", i+'a');
    }
    printf("\n");

    for(int i = 0; i < SIZE; i++) {
        printf("%2d ", i+1);
        for(int j = 0; j < SIZE; j++) {
            printf("%d ", state->board[i][j]);
        }
        printf("\n");
    }
}

void add_stone(GameState *state, char column, int row) {
    int x = column - 'a';
    int y = row - 1;

    if (x >= 0 && x < SIZE && y >= 0 && y < SIZE && state->board[y][x] == EMPTY) {
        state->board[y][x] = state->current_stone;
        // state->current_stone = (state->current_stone == BLACK) ? WHITE : BLACK;
    }
}

bool add_stone_computer(GameState *state) {
    for (int attempts = 0; attempts < SIZE * SIZE; attempts++) {
        int x = rand() % SIZE;
        int y = rand() % SIZE;
        if (state->board[y][x] == EMPTY) {
            state->board[y][x] = state->current_stone;
            // state->current_stone = (state->current_stone == BLACK) ? WHITE : BLACK;
            return true;
        }
    }

    return false;  // no empty cells were found after a certain number of attempts
}

