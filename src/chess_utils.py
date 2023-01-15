from src import process_data as pdata

def unpack_moves(pgn_str):
    ''' 
    Returns a list of tuples, where each tuples is 
    (move, evaluation at move)
    '''
    moves = pgn_str.split('. ')[1:]
    unpacked_moves = []
    for move_str in moves:
        move = pdata.get_substring('', ' {', move_str)
        eval = float(pdata.get_substring('eval ', '] ', move_str))
        unpacked_moves.append((move, eval))

    return unpacked_moves

def unpack_moves_df(pgn_df):
    pgn_df['unpacked_moves'] = pgn_df.apply(lambda row: unpack_moves(row['pgn']), axis = 1)
    return pgn_df

def attribute_moves_df(pgn_df, unpack = False):
    if unpack:
        pgn_df = unpack_moves_df(pgn_df)

    pgn_df['game_attributes'] = pgn_df.apply(lambda row: get_move_attributes_game(row['unpacked_moves']), axis = 1)

    return pgn_df

def get_square(move_str):
    for i, c in enumerate(move_str):
        if c.isdigit():
            return move_str[i-1:i+1]
    
    return '!' 

def get_move_attributes(move_tuple):
    move = move_tuple[0]
    if move == None:
        return ('!', 0, 0, 0)

    capture = int('x' in move)
    check = int(('+' in move) or ('#' in move))
    pawn_density = 0 

    piece = move[0]
    if piece not in ['K', 'Q', 'R', 'B', 'N']:
        piece = 'P'

    square = get_square(move)

    central_squares = ['c3', 'c4', 'c5', 'c6', 
                       'd3', 'd4', 'd5', 'd6',  
                       'e3', 'e4', 'e5', 'e6',
                       'f3', 'f4', 'f5', 'f6']

    if piece == 'P' and square in central_squares:
        pawn_density += 1

    return (piece, capture, check, pawn_density)


def get_move_attributes_game(move_lst):
    piece_movements = {'K': 0, 'Q': 0, 'R': 0, 'B': 0, 'N': 0, 'P': 0}
    captures = 0
    checks = 0
    pawn_density = 0

    attributes_5 = []
    attributes_10 = []
    attributes_15 = []
    attributes_final = []

    for move_num, move in enumerate(move_lst):
        piece, capture, check, pawn_density = get_move_attributes(move)

        if piece == '!':
            continue 

        piece_movements[piece] += 1
        captures += capture
        checks += check
        pawn_density += pawn_density

        if move_num == 4:
            attributes_5 = [piece_movements[piece] for piece in piece_movements.keys()]
            attributes_5 += [captures, checks, pawn_density]
        if move_num == 9:
            attributes_10 = [piece_movements[piece] for piece in piece_movements.keys()]
            attributes_10 += [captures, checks, pawn_density]
        if move_num == 14:
            attributes_15 = [piece_movements[piece] for piece in piece_movements.keys()]
            attributes_15 += [captures, checks, pawn_density]

    attributes_final = [piece_movements[piece] for piece in piece_movements.keys()]
    attributes_final += [captures, checks, pawn_density]

    return attributes_5, attributes_10, attributes_15, attributes_final
