#ifndef BOARD_H
#define BOARD_H

#include <stdbool.h>

#define SIZE 7
#define EMPTY 0
#define BLACK 1
#define WHITE 2

extern int board[SIZE][SIZE];
extern int current_stone;

void initialize_board();
void print_board();
void add_stone(char column, int row, int current_stone);
bool add_stone_computer();
bool is_valid_move();
int get_valid_moves();
#endif /* BOARD_H */

