int checkWin(int player) {
    // Check rows
    for (int row = 0; row < BOARD_SIZE; row++) {
        for (int col = 0; col <= BOARD_SIZE - 5; col++) {
            int count = 0;
            for (int i = 0; i < 5; i++) {
                if (board[row][col + i] == player)
                    count++;
                else
                    break;
            }
            if (count == 5)
                return 1;
        }
    }

    // Check columns
    for (int col = 0; col < BOARD_SIZE; col++) {
        for (int row = 0; row <= BOARD_SIZE - 5; row++) {
            int count = 0;
            for (int i = 0; i < 5; i++) {
                if (board[row + i][col] == player)
                    count++;
                else
                    break;
            }
            if (count == 5)
                return 1;
        }
    }

    // Check diagonals (top-left to bottom-right)
    for (int row = 0; row <= BOARD_SIZE - 5; row++) {
        for (int col = 0; col <= BOARD_SIZE - 5; col++) {
            int count = 0;
            for (int i = 0; i < 5; i++) {
                if (board[row + i][col + i] == player)
                    count++;
                else
                    break;
            }
            if (count == 5)
                return 1;
        }
    }

    // Check diagonals (top-right to bottom-left)
    for (int row = 0; row <= BOARD_SIZE - 5; row++) {
        for (int col = BOARD_SIZE - 1; col >= 4; col--) {
            int count = 0;
            for (int i = 0; i < 5; i++) {
                if (board[row + i][col - i] == player)
                    count++;
                else
                    break;
            }
            if (count == 5)
                return 1;
        }
    }

    return 0;
}
