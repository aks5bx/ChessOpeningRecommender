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
