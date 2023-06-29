#include <stdio.h>
#include <stdlib.h>
#include "board.h"
#include "winfunc.h"
#include "mcts_mpi.h"
#include <mpi.h>

int TERMINATE_TAG = 77;
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, n_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

    char column;
    int row;
    int your_color;

    GameState state;
    if (rank ==0){
        initialize_board(&state);
        your_color = 1; //you are black stone
    }
    while(1) {
        if (rank==0){
            if ((state.current_stone == your_color)) {
                print_board(&state);
                printf("Enter coordinates (e.g., h8): ");
                scanf(" %c%d", &column, &row);
                add_stone(&state, column, row);
        	    state.current_stone = (state.current_stone == BLACK) ? WHITE : BLACK;
            } 
            else {
                //printf("agin in else loop in rank 0 \n");
                int max_search = 10000;
                // int best_action = run_mcts(state, max_search);
                //print_board(&state);
                int possible_action_num = get_valid_move_count(&state);
                int wins[possible_action_num], visits[possible_action_num];
                
                for (int i = 1; i < n_proc; i++) {
                    //MPI_Barrier(MPI_COMM_WORLD);
                    int terminate = 0;
                    MPI_Send(&terminate, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                    MPI_Send(&state, sizeof(GameState), MPI_BYTE, i, 0, MPI_COMM_WORLD);
                    MPI_Send(&max_search, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                    MPI_Status status;
                    int** slave_result = calloc(2, sizeof(int*));
                    for (int i = 0; i < 2; i++) {
                        slave_result[i] = malloc(sizeof(int));
                    }
                   
                    //MPI_Barrier(MPI_COMM_WORLD);
                    MPI_Recv(&(slave_result[0][0]), 2*possible_action_num, MPI_INT, i, 34, MPI_COMM_WORLD, &status);
                    MPI_Recv(&(slave_result[1][0]), 2*possible_action_num, MPI_INT, i, 33, MPI_COMM_WORLD, &status);
                    printf("master recieve from source :%d, tag :%d\n Recived result was %d\n", status.MPI_SOURCE, status.MPI_TAG, slave_result[1][2]);
                    //printf("error code: %d\n", status.MPI_ERROR);
                    //printf("something unrelate to slave_result\n");

                    //printf("something\n");
                
                // Add the received wins and visits arrays to the respective arrays in the master node
                    for (int j = 0; j < possible_action_num; j++) {
                        wins[j] += slave_result[0][j];
                        visits[j] += slave_result[1][j];
                    }
                    free(slave_result[0]);
                    free(slave_result[1]);
                }

                // impelement wins/visits to find best action
                int action = 0; int score = 0;
                for (int k =1; k < possible_action_num; k++){
                    if (visits[k] == 0){
                        continue;
                    }
                    if (wins[k]/visits[k]>score){
                        score = wins[k]/visits[k];
                        action = k;
                    }    
                }
                printf("wins : %d , visits : %d", wins[action], visits[action]);
                printf("this move has score = %d\n", score);

                //free(slave_result[0]);
                //free(slave_result[1]);


                make_move(&state, action); //best_action = action
                //add_stone_computer(&state); 
            }
            if (checkWinDebug(&state)){
                print_board(&state);
                printf("%d win the game\n", 3 - state.current_stone);

                // Terminate slaves
                for (int i = 1; i < n_proc; i++) {
                    int terminate = 1;
                    MPI_Send(&terminate, 1, MPI_INT, i, TERMINATE_TAG, MPI_COMM_WORLD);
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
                    MPI_Send(&terminate, 1, MPI_INT, i, TERMINATE_TAG, MPI_COMM_WORLD);
                } 
 
            }
            continue;
        }
        else {  // Slave processes
            GameState state;
            int max_search;
            int dummy;
            MPI_Status status;
            //printf("node %d listening\n", rank);
            MPI_Recv(&dummy, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TERMINATE_TAG){
                break;
            }
            MPI_Recv(&state, sizeof(GameState), MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&max_search, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int possible_action_num = get_valid_move_count(&state);
            int** result = malloc(2 * sizeof(int*));
            for (int i = 0; i < 2; i++) {
                result[i] = malloc(sizeof(int));
            }

            result = monte_carlo_tree_search(state, max_search); // Update visits directly without redeclaration
            printf("Before sending : %d\n", result[1][2]);
            MPI_Send(&(result[0][0]), 2*possible_action_num, MPI_INT, 0, 34, MPI_COMM_WORLD);
            MPI_Send(&(result[1][0]), 2*possible_action_num, MPI_INT, 0, 33, MPI_COMM_WORLD);
            free(result[0]);
            free(result[1]);
            //printf("send!\n");
            //MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
    return 0;
}

