#include <stdio.h>
#include "board.h"

int main() {
    char column;
    int row;
    
    initialize_board();
    
    while(1) {
        print_board();
        printf("Enter coordinates (e.g., h8): ");
        scanf(" %c%d", &column, &row);
        add_stone(column, row);
    }
    
    return 0;
}

