#include <stdio.h>
#include <stdlib.h>
#include "board.h"
#include "winfunc.h"
#include "mcts_mpi.h"
#include <mpi.h>

#define TERMINATE_TAG 0

void slave(){

}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, n_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

    char column;
    int row;
    int your_color;

    GameState state;

    if (rank == 0){
    initialize_board(&state);
    your_color = 1; //you are black stone
    }

    if (rank==0){
        while(1) {
            if ((state.current_stone == your_color)) {
                print_board(&state);
                printf("Enter coordinates (e.g., h8): ");
                scanf(" %c%d", &column, &row);
                add_stone(&state, column, row);
        	    state.current_stone = (state.current_stone == BLACK) ? WHITE : BLACK;
            } 
            else {
                int max_search = 10000;
                for (int i = 1; i < n_proc; i++) {
                    MPI_Send(&state, sizeof(GameState), MPI_BYTE, i, 0, MPI_COMM_WORLD);
                    MPI_Send(&max_search, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                }
                // int best_action = run_mcts(state, max_search);
                //print_board(&state);
                int action;
                int possible_action_num = get_valid_move_count(&state);
                int wins[possible_action_num], visits[possible_action_num];

                for (int i = 1; i < n_proc; i++) {
                    int slave_wins[possible_action_num], slave_visits[possible_action_num];
                    MPI_Recv(&slave_wins, possible_action_num, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&slave_visits, possible_action_num, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Add the received wins and visits arrays to the respective arrays in the master node
                    for (int j = 0; j < possible_action_num; j++) {
                        wins[j] += slave_wins[j];
                        visits[j] += slave_visits[j];
                    }
                }

                // impelement wins/visits to find best action

                make_move(&state, action); //best_action = action
                //add_stone_computer(&state); 
            }
            if (checkWinDebug(&state)){
                print_board(&state);
                printf("%d win the game\n", 3 - state.current_stone);

                // Terminate slaves
                for (int i = 1; i < n_proc; i++) {
                    int terminate = 1;
		    for (int i = 1; i < n_proc; i++) {
 			   MPI_Send(&terminate, 1, MPI_INT, i, TERMINATE_TAG, MPI_COMM_WORLD);
			}
                } 
                break;
            }
            if (checkDraw(&state)){
                print_board(&state);
                printf("There is no valid place - Draw");
                break;
                // Terminate slaves
                for (int i = 1; i < n_proc; i++) {
                    int terminate = 1;
                    MPI_Send(&terminate, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                } 
 
            }
        }
    }  
    else {  // Slave processes
        int flag = 0; // initialize flag to zero
        
        while (!flag) {
            MPI_Status status;
            MPI_Iprobe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        
            if (flag) {
                if (status.MPI_TAG == TERMINATE_TAG) {
                    printf("Received Terminate signal from Master Node\n");
                    
                    // Perform necessary cleanup or termination actions
                    
                    break; // Exit the while loop
                } else if (status.MPI_TAG == 0) {
                    GameState state;
                    int max_search;
                    MPI_Recv(&state, sizeof(GameState), MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&max_search, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    int possible_action_num = get_valid_move_count(&state);
                    int action;
                    int wins[possible_action_num];
                    int visits[possible_action_num];
                    monte_carlo_tree_search(state, max_search, &wins, &visits); // Update visits directly without redeclaration
                    
                    MPI_Send(&wins, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
                }
            }
        }
        
    }
        MPI_Finalize();
    return 0;
}

