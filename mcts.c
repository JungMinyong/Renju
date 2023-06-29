#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "winfunc.h"
#include "board.h"

int monte_carlo_tree_search(GameState *state) {
    int num_actions = get_valid_move_count(state);
    int wins[num_actions];
    int visits[num_actions];
    for (int i = 0; i < num_actions; i++) {
        wins[i] = 0;
        visits[i] = 0;
    }

    for (int i = 0; i < 1000; i++) {
        int action = rand() % num_actions;
        GameState new_state = make_move(state, action);
	int winner = simulate_game(new_state);
        visits[action]++;
        if (state->current_stone == winner) {    
	    wins[action]++;
        }
    }

    int best_action = 0;
    double best_ratio = -1;
    for (int i = 0; i < num_actions; i++) {
        double ratio = (double)wins[i] / visits[i];
        if (ratio > best_ratio) {
            best_ratio = ratio;
            best_action = i;
        }
    }

    return best_action;
}

int simulate_game(GameState state) {
    GameState copy_state = state; // making a copy to not alter the original state

    while (!is_game_over(&copy_state)) {
        int num_actions = get_valid_move_count(&copy_state);
        if (num_actions == 0) {
            break;
        }
        int action = rand() % num_actions;
        copy_state = make_move(&copy_state, action);
    }

    return get_winner(&copy_state);
}

