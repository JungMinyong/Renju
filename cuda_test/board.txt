#ifndef BOARD_H
#define BOARD_H

#include <stdbool.h>

#define SIZE 7
#define EMPTY 0
#define BLACK 1
#define WHITE 2

extern int board[SIZE][SIZE];
extern int current_stone;

typedef struct {
    int row;
    int col;
} Move;

typedef struct {
    int current_stone;
    int last_move;
    int board[SIZE][SIZE];
} GameState;

typedef struct {
    int count;
    int rows[SIZE*SIZE];
    int cols[SIZE*SIZE];
} ValidMoves;

void initialize_board(GameState* state);
void print_board(GameState* state);
void add_stone(GameState* state, char column, int row);

bool add_stone_computer(GameState* state);
bool is_valid_move(GameState* state, int row, int col);
ValidMoves get_valid_moves(GameState* state);
#endif /* BOARD_H */

