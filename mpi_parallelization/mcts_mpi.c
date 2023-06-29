#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "winfunc.h"
#include "board.h"

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


int monte_carlo_tree_search(GameState original_state, int MAX_SEARCH) { // now takes GameState instead of GameState *
    GameState state = original_state;
    int num_actions = get_valid_move_count(&state);
    int current_stone = state.current_stone;
    int wins[num_actions];
    int visits[num_actions];
    for (int i = 0; i < num_actions; i++) {
        wins[i] = 0;
        visits[i] = 0;
    }

    for (int i = 0; i < MAX_SEARCH; i++) {
        int action = rand() % num_actions;
        GameState new_state = make_move_copy(state, action);
	int winner = simulate_game(new_state);
        visits[action]++;
        if (current_stone == winner) { // accessing the value directly
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

    return best_action, wins, visits;
}

