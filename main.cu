#include <stdio.h>
#include <stdlib.h>
#include "board.h"
#include "ai.h"  // Include the header for the AI code

int main() {
    char column;
    int row;
    char your_color;

    initialize_board();
    your_color = BLACK; //you are black stone

    while(1) {
        print_board();
        if (current_stone == your_color) {
            printf("Enter coordinates (e.g., h8): ");
            scanf(" %c%d", &column, &row);
            add_stone(column, row, current_stone);
        } else {
            add_stone_computer(); // Call the function defined in ai.cu
        }
        current_stone = (current_stone == BLACK) ? WHITE : BLACK; //flip the color of the stone
    }
    return 0;
}

