#ifndef BOARD_H
#define BOARD_H

#include <stdbool.h>

#define SIZE 15
#define EMPTY '.'
#define BLACK 'X'
#define WHITE 'O'

extern char board[SIZE][SIZE];
extern char current_stone;

void initialize_board();
void print_board();
void add_stone(char column, int row, char current_stone);
bool add_stone_computer();

#endif /* BOARD_H */

