#include "board.h"


bool checkWinDebug(GameState *state);
bool checkWin(GameState *state);
bool checkDraw(GameState *state);

bool checkDraw(GameState *state);
int get_winner(GameState* state);
int is_game_over(GameState* state);
int get_valid_move_count(GameState* state);

GameState make_move_copy(GameState state, int action);
GameState make_move(GameState* state, int action);
