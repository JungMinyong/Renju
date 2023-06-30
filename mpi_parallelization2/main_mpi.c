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
    int terminate = 0;
    MPI_Status status;

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
                    terminate = 0;

                    MPI_Send(&terminate, 1, MPI_INT, i, TERMINATE_TAG, MPI_COMM_WORLD);
                    MPI_Send(&state, sizeof(GameState), MPI_BYTE, i, 0, MPI_COMM_WORLD);
                    MPI_Send(&max_search, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                    MPI_Status status;
                }
                    
                printf("master, finish sending");
                 for (int i = 1; i < n_proc; i++){
                    int slave_wins[possible_action_num];
                    int slave_visits[possible_action_num];
                      
                    MPI_Recv(slave_wins, possible_action_num, MPI_INT, i, 34, MPI_COMM_WORLD, &status);
                    MPI_Recv(slave_visits, possible_action_num, MPI_INT, i, 33, MPI_COMM_WORLD, &status);

                    printf("master recieve from source :%d, tag :%d\n Recived result was %d\n", status.MPI_SOURCE, status.MPI_TAG, slave_visits[2]);
                    
                    for (int j = 0; j < possible_action_num; j++) {
                        wins[j] += slave_wins[j];
                        visits[j] += slave_visits[j];
                        }
                     
                 }
                   
                    //MPI_Barrier(MPI_COMM_WORLD);
                    //MPI_Recv(&(slave_result[0][0]), 2*possible_action_num, MPI_INT, i, 34, MPI_COMM_WORLD, &status);
                    //MPI_Recv(&(slave_result[1][0]), 2*possible_action_num, MPI_INT, i, 33, MPI_COMM_WORLD, &status);
                    
               
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
            printf("slave start %d \n", rank);
            GameState state;
            int max_search;
            //int dummy;
            MPI_Status status;
            //printf("node %d listening\n", rank);
            MPI_Recv(&terminate, 1, MPI_INT, 0, TERMINATE_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == 1){
                printf("break slave");
                break;
            }
            MPI_Recv(&state, sizeof(GameState), MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&max_search, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            
            int possible_action_num = get_valid_move_count(&state);
            int *wins_chunk = malloc(possible_action_num * sizeof(int));
            int *visits_chunk = malloc(possible_action_num * sizeof(int));            //int wins_chunk = NULL;
            //int visits_chunk = NULL;

            printf("possible_action_num %d \n", possible_action_num);
            monte_carlo_tree_search(state, max_search, wins_chunk, visits_chunk);           
            printf("Before sending : %d\n", visits_chunk[2]);
            //MPI_Send(&(result[0][0]), 2*possible_action_num, MPI_INT, 0, 34, MPI_COMM_WORLD);
            //MPI_Send(&(result[1][0]), 2*possible_action_num, MPI_INT, 0, 33, MPI_COMM_WORLD);
            MPI_Send(wins_chunk, possible_action_num, MPI_INT, 0, 34, MPI_COMM_WORLD);
            MPI_Send(visits_chunk, possible_action_num, MPI_INT, 0, 33, MPI_COMM_WORLD);

            free(wins_chunk);
            free(visits_chunk);
            printf("slave %d end! \n", rank);
            

        }
    }
    MPI_Finalize();
    return 0;
}

