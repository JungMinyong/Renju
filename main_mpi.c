#include <stdio.h>
#include <stdlib.h>
#include "board.h"
#include "winfunc.h"
#include "mcts.h"
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char column;
    int row;
    int your_color;

    GameState state;

    if (rank == 0){
initialize_board(&state);
    your_color = 1; //you are black stone

    while(1) {
        if (state.current_stone == your_color) {
            print_board(&state);
            printf("Enter coordinates (e.g., h8): ");
            scanf(" %c%d", &column, &row);
            add_stone(&state, column, row);
    	    state.current_stone = (state.current_stone == BLACK) ? WHITE : BLACK;
        } else {
            int max_search = 10000;
    	    int best_action = monte_carlo_tree_search(state, max_search);
    	    print_board(&state);
    	    make_move(&state, best_action);
    	    //add_stone_computer(&state);
                }
        if (checkWin(&state)){
            print_board(&state);
            printf("%d win the game\n", state.current_stone);
            break;
            }
        if (checkDraw(&state)){
            print_board(&state);
            printf("There is no valid place - Draw");
            break;
            }
        }
        
    }
        MPI_Finalize();
    return 0;
}
