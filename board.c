#include <stdio.h>
#include "board.h"

int board[SIZE][SIZE];
int current_stone = 1; //BLACK;

void initialize_board() {
    int i;
    int j;
    for(i = 0; i < SIZE; i++) {
        for(j = 0; j < SIZE; j++) {
            board[i][j] = EMPTY;
        }
    }
}

bool is_valid_move(row, col){
    return board[row][col] == EMPTY;
}

int get_valid_moves(){
    int valid_moves_row[SIZE*SIZE];
    int valid_moves_col[SIZE*SIZE];
    int i;
    int j;
    for (i = 0; i < SIZE; i++) {
	for (j = 0; j < SIZE, j++){
	    if (is_valid_move(i,j)){
		valid_moves_row[i + j*SIZE] = i;
		valid_moves_col[i + j*SIZE] = j;
	    }
	}
    }
    return valid_moves_row, valid_moves_col;
}
//print current board status
void print_board() {
    printf("   ");
    int i;
    int j;
    for(i = 0; i < SIZE; i++) {
        printf("%c ", i+'a');
    }
    printf("\n");

    for(i = 0; i < SIZE; i++) {
        printf("%2d ", i+1);
        for(j = 0; j < SIZE; j++) {
            printf("%d ", board[i][j]);
        }
        printf("\n");
    }
}

//add stone manually. 
void add_stone(char column, int row, int current_stone) {
    int x = column - 'a';
    int y = row - 1;
    
    if (x >= 0 && x < SIZE && y >= 0 && y < SIZE && board[y][x] == EMPTY) {
        board[y][x] = current_stone;
        //current_stone = (current_stone == BLACK) ? WHITE : BLACK;
    }
}

bool add_stone_computer() {
    int attempts;
    for (attempts = 0; attempts < SIZE * SIZE; attempts++) {
        int x = rand() % SIZE;
        int y = rand() % SIZE;
        if (board[y][x] == EMPTY) {
            board[y][x] = current_stone;
            //current_stone = (current_stone == BLACK) ? WHITE : BLACK;
            return true;
        }
    }

    return false;  // no empty cells were found after a certain number of attempts
}
