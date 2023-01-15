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

def prepare_final_df(games_df):
    games_df['move5'], games_df['move10'], games_df['move15'], games_df['final'], games_df['opening_eval'] = zip(*games_df['game_attributes'])
    games_df['move5_K'], games_df['move5_Q'], games_df['move5_R'], games_df['move5_B'], games_df['move5_N'], games_df['move5_P'], games_df['move5_captures'], games_df['move5_checks'], games_df['move5_pawn_density'] = zip(*games_df['move5'])
    games_df['move10_K'], games_df['move10_Q'], games_df['move10_R'], games_df['move10_B'], games_df['move10_N'], games_df['move10_P'], games_df['move10_captures'], games_df['move10_checks'], games_df['move10_pawn_density'] = zip(*games_df['move10'])
    games_df['move15_K'], games_df['move15_Q'], games_df['move15_R'], games_df['move15_B'], games_df['move15_N'], games_df['move15_P'], games_df['move15_captures'], games_df['move15_checks'], games_df['move15_pawn_density'] = zip(*games_df['move15'])

    feature_df = games_df[['user_name', 'user_elo', 'opening_code', 'opening_name', 'opening_category', 'opening_eval', 'move5_K', 'move5_Q', 'move5_R', 'move5_B', 'move5_N', 'move5_P', 'move5_captures', 'move5_checks', 'move5_pawn_density', 'move10_K', 'move10_Q', 'move10_R', 'move10_B', 'move10_N', 'move10_P', 'move10_captures', 'move10_checks', 'move10_pawn_density', 'move15_K', 'move15_Q', 'move15_R', 'move15_B', 'move15_N', 'move15_P', 'move15_captures', 'move15_checks', 'move15_pawn_density']]
    
    return feature_df

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

    attributes_5 = [0,0,0,0,0,0,0,0,0]
    attributes_10 = [0,0,0,0,0,0,0,0,0]
    attributes_15 = [0,0,0,0,0,0,0,0,0]
    attributes_final = []

    if len(move_lst) < 19 and len(move_lst) % 2 == 0:
        last_3_evals = [move_lst[-6][1],move_lst[-4][1],move_lst[-2][1]]
    elif len(move_lst) < 19 and len(move_lst) % 2 != 0:
        last_3_evals = [move_lst[-5][1],move_lst[-3][1],move_lst[-1][1]]
    else:
        last_3_evals = [move_lst[14][1],move_lst[16][1],move_lst[18][1]]

    last_3_evals = [x for x in last_3_evals if x is not None]
    opening_eval = round(sum(last_3_evals) / len(last_3_evals), 2)
    
    for move_num, move in enumerate(move_lst):

        if move_num % 2 != 0:
            continue
        else:
            move_num = (move_num / 2) + 1

        piece, capture, check, pawn_density = get_move_attributes(move)

        if piece == '!':
            continue 

        piece_movements[piece] += 1
        captures += capture
        checks += check
        pawn_density += pawn_density

        if move_num == 5:
            attributes_5 = [piece_movements[piece] for piece in piece_movements.keys()]
            attributes_5 += [captures, checks, pawn_density]
        if move_num == 10:
            attributes_10 = [piece_movements[piece] for piece in piece_movements.keys()]
            attributes_10 += [captures, checks, pawn_density]
        if move_num == 15:
            attributes_15 = [piece_movements[piece] for piece in piece_movements.keys()]
            attributes_15 += [captures, checks, pawn_density]

    attributes_final = [piece_movements[piece] for piece in piece_movements.keys()]
    attributes_final += [captures, checks, pawn_density]

    try:
        return attributes_5, attributes_10, attributes_15, attributes_final, opening_eval
    except:
        print(last_3_evals)
        print(move_lst)
        return attributes_5, attributes_10, attributes_15, attributes_final, opening_eval
