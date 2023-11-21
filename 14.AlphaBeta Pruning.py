import math

# Function to check if a player has won the game
def check_win(board, player):
    win_conditions = [
        [board[0], board[1], board[2]],
        [board[3], board[4], board[5]],
        [board[6], board[7], board[8]],
        [board[0], board[3], board[6]],
        [board[1], board[4], board[7]],
        [board[2], board[5], board[8]],
        [board[0], board[4], board[8]],
        [board[2], board[4], board[6]],
    ]
    for condition in win_conditions:
        if condition.count(player) == 3:
            return True
    return False

# Function to check if the game is over
def game_over(board):
    return check_win(board, 'X') or check_win(board, 'O') or ' ' not in board

# Function to get all possible moves
def get_possible_moves(board):
    return [i for i, mark in enumerate(board) if mark == ' ']

# Alpha-Beta pruning algorithm implementation
def alpha_beta(board, depth, alpha, beta, is_maximizing):
    if check_win(board, 'X'):
        return -10 + depth
    elif check_win(board, 'O'):
        return 10 - depth
    elif ' ' not in board:
        return 0

    if is_maximizing:
        best_score = -math.inf
        for move in get_possible_moves(board):
            board[move] = 'O'
            score = alpha_beta(board, depth + 1, alpha, beta, False)
            board[move] = ' '
            best_score = max(score, best_score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score
    else:
        best_score = math.inf
        for move in get_possible_moves(board):
            board[move] = 'X'
            score = alpha_beta(board, depth + 1, alpha, beta, True)
            board[move] = ' '
            best_score = min(score, best_score)
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_score

# Function to determine the best move using Alpha-Beta pruning
def best_move(board):
    best_score = -math.inf
    best_move = None
    alpha = -math.inf
    beta = math.inf
    for move in get_possible_moves(board):
        board[move] = 'O'
        score = alpha_beta(board, 0, alpha, beta, False)
        board[move] = ' '
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, best_score)
    return best_move

# Function to display the Tic Tac Toe board
def display_board(board):
    print('-------------')
    for i in range(0, 9, 3):
        print(f'| {board[i]} | {board[i + 1]} | {board[i + 2]} |')
        print('-------------')

# Main function to play the game
def play_tic_tac_toe():
    board = [' '] * 9
    while not game_over(board):
        display_board(board)
        player_move = int(input('Enter your move (0-8): '))
        if board[player_move] == ' ':
            board[player_move] = 'X'
        else:
            print('Invalid move, try again.')
            continue

        if not game_over(board):
            computer_move = best_move(board)
            board[computer_move] = 'O'

    display_board(board)
    if check_win(board, 'X'):
        print('You win!')
    elif check_win(board, 'O'):
        print('Computer wins!')
    else:
        print('It\'s a draw!')

# Play the game
play_tic_tac_toe()
