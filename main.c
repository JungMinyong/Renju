#include <stdio.h>
#include "board.h"

int main() {
    char column;
    int row;
    
    initialize_board();
    
    while(1) {
        print_board();
        if (current_stone == BLACK) {
            printf("Enter coordinates (e.g., h8): ");
            scanf(" %c%d", &column, &row);
            add_stone(column, row);
        } else {
            add_stone_computer();
        }
    } 
    return 0;
}

