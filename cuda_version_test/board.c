#include <stdio.h>
#include "board.h"

char board[SIZE][SIZE];
char current_stone = BLACK;

void initialize_board() {
    for(int i = 0; i < SIZE; i++) {
        for(int j = 0; j < SIZE; j++) {
            board[i][j] = EMPTY;
        }
    }
}

//print current board status
void print_board() {
    printf("   ");
    for(int i = 0; i < SIZE; i++) {
        printf("%c ", i+'a');
    }
    printf("\n");

    for(int i = 0; i < SIZE; i++) {
        printf("%2d ", i+1);
        for(int j = 0; j < SIZE; j++) {
            printf("%c ", board[i][j]);
        }
        printf("\n");
    }
}

//add stone manually. 
void add_stone(char column, int row, char current_stone) {
    int x = column - 'a';
    int y = row - 1;
    
    if (x >= 0 && x < SIZE && y >= 0 && y < SIZE && board[y][x] == EMPTY) {
        board[y][x] = current_stone;
        //current_stone = (current_stone == BLACK) ? WHITE : BLACK;
    }
}

bool add_stone_computer() {
    for (int attempts = 0; attempts < SIZE * SIZE; attempts++) {
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
