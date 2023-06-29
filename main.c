#include <stdio.h>
#include <stdlib.h>
#include "board.h"
#include "winfunc.h"
#include "mcts.h"

int main(int argc, char *argv[]) {
    int rank;

    char column;
    int row;
    int your_color;

    GameState state;
    initialize_board(&state);
    your_color = 1; //you are black stone


    bool gameOver = false;
   
    while (!gameOver) {
        if (state.current_stone == your_color) {
            print_board(&state);
            printf("Enter coordinates (e.g., h8): ");
            scanf(" %c%d", &column, &row);
            add_stone(&state, column, row);
            state.current_stone = (state.current_stone == BLACK) ? WHITE : BLACK;
        } else {
            int max_search = 10000;
            int best_action = monte_carlo_tree_search(state, max_search);
            make_move(&state, best_action);
        }
   
   
        if (checkWinDebug(&state)) {
            print_board(&state);
            printf("Player %d wins the game\n", 3 - state.current_stone);
            printf("Congratulations!\n");
            gameOver = true;
        } else if (checkDraw(&state)) {
            print_board(&state);
            printf("There is no valid place - Draw");
            gameOver = true;
        }
   
    }
   
     return 0;
}
