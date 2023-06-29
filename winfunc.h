#include "board.h"


bool checkWin(GameState *state);
bool checkDraw(GameState *state);

bool checkDraw(GameState *state);
int get_winner(GameState* state);
int is_game_over(GameState* state);
int get_valid_move_count(GameState* state);

GameState make_move(GameState* state, int action);
