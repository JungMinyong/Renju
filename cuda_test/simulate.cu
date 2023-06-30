#include "simulate.h"


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