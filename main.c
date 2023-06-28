#include <stdio.h>
#include <stdlib.h>
#include "board.h"
#include "winfunc.h"

int main() {
    char column;
    int row;
    int your_color;

    initialize_board();
    your_color = 1; //you are black stone

    while(1) {
        print_board();
        if (current_stone == your_color) {
            printf("Enter coordinates (e.g., h8): ");
            scanf(" %c%d", &column, &row);
            add_stone(column, row, current_stone);
        } else {
            add_stone_computer();
        }
        if (checkWin(current_stone)){
	    print_board();
	    printf("%d win the game\n", current_stone);
	    break;
	}
	current_stone = (current_stone == 1) ? 2 : 1; //flip the color of the stone
    } 
    return 0;
}

