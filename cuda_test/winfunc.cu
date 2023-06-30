#include "board.h"

int BOARD_SIZE = SIZE;

bool checkWin(GameState *state) {
    int player = state->current_stone;
	// Check rows
    for (int row = 0; row < BOARD_SIZE; row++) {
        for (int col = 0; col <= BOARD_SIZE - 5; col++) {
            int count = 0;
            for (int i = 0; i < 5; i++) {
                if (state->board[row][col + i] == player)
                    count++;
                else
                    break;
            }
            if (count == 5)
                return true;
        }
    }

    // Check columns
    for (int col = 0; col < BOARD_SIZE; col++) {
        for (int row = 0; row <= BOARD_SIZE - 5; row++) {
            int count = 0;
            for (int i = 0; i < 5; i++) {
                if (state->board[row + i][col] == player)
                    count++;
                else
                    break;
            }
            if (count == 5)
                return true;
        }
    }

    // Check diagonals (top-left to bottom-right)
    for (int row = 0; row <= BOARD_SIZE - 5; row++) {
        for (int col = 0; col <= BOARD_SIZE - 5; col++) {
            int count = 0;
            for (int i = 0; i < 5; i++) {
                if (state->board[row + i][col + i] == player)
                    count++;
                else
                    break;
            }
            if (count == 5)
                return true;
        }
    }

    // Check diagonals (top-right to bottom-left)
    for (int row = 0; row <= BOARD_SIZE - 5; row++) {
        for (int col = BOARD_SIZE - 1; col >= 4; col--) {
            int count = 0;
            for (int i = 0; i < 5; i++) {
                if (state->board[row + i][col - i] == player)
                    count++;
                else
                    break;
            }
            if (count == 5)
                return true;
        }
    }

    return false;
}

bool checkDraw(GameState *state){
    for (int row = 0; row < BOARD_SIZE; row++) {
        for (int col = 0; col < BOARD_SIZE; col++) {
            if (state->board[row][col] == 0){
                return false;
            }       
        }
    }
    return true;
}

 
int get_winner(GameState* state) {
    return checkWin(state) ? state->current_stone : -1;
}

int is_game_over(GameState* state) {
    return checkWin(state) || checkDraw(state);
}


int get_valid_move_count(GameState* state) {
    return get_valid_moves(state).count;
}


GameState make_move(GameState* state, int action) {
    ValidMoves valid_moves = get_valid_moves(state);
    int row = valid_moves.rows[action];
    int col = valid_moves.cols[action];
    add_stone(state, col + 'a', row + 1);
    state->current_stone = (state->current_stone == BLACK) ? WHITE : BLACK;
    return *state;
}

GameState make_move_copy(GameState state, int action) {
    ValidMoves valid_moves = get_valid_moves(&state);
    int row = valid_moves.rows[action];
    int col = valid_moves.cols[action];
    add_stone(&state, col + 'a', row + 1);
    state.current_stone = (state.current_stone == BLACK) ? WHITE : BLACK;
    return state;
}

