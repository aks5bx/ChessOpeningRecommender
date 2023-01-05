from itertools import islice
import re
import json
import time 

def get_substring(start_char, end_char, text):
    '''
    Helper function for unpack_game()
    '''

    search_string = start_char + '(.+?)' + end_char
    result = re.search(search_string, text)

    if result:
        return result.group(1)
    else:
        return None 

def unpack_game(data_lst):
    '''
    Takes a list of lines and returns a dictionary of game data
    '''

    if ('White' not in data_lst[4]) or ('WhiteElo' not in data_lst[9]):
        return None
        #print('ERROR')
        #print('')
        #print(data_lst)
        #print('')
        #print(data_lst[4])

    username_w = get_substring('\"', '\"', data_lst[4])
    userelo_w = get_substring('\"', '\"', data_lst[9])

    opening_code = get_substring('\"', '\"', data_lst[13])
    opening_name = get_substring('\"', '\"', data_lst[14])

    pgn = data_lst[18]

    return {'user_name': username_w, 'user_elo': userelo_w, 'opening_code': opening_code, 'opening_name': opening_name, 'pgn': pgn}

def load_games(cfg):
    '''
    Loads games from a pgn file and returns a dictionary of games 
    '''

    ## Open cfg 
    filename = cfg['live_run']['dataset_file'] 
    dataset = cfg['live_run']['dataset']
    outfile_name = 'data/processed_' + dataset + '.json'

    ## Define game dictionary and open file 
    game_dict = {}
    with open(filename) as f:
        game_id = 1
        i = 0
        program_starts = time.time()
        while True:

            ## Track the while loop
            if i % 5000000 == 0:
                print(round(((i * 20) /cfg[dataset]['num_lines']) * 100, 2), '% Complete')
                now = time.time()
                print('Time Elapsed ', round(now - program_starts, 2), ' seconds')
                print('')
            i += 1
            ## Grab 20 lines at a time (this is one game)
            next_n_lines = list(islice(f, 20))
            ## Stopping condition for while loop 
            if not next_n_lines or len(next_n_lines) < 19:
                break
            ## Don't include games without eval data 
            if 'eval' not in next_n_lines[18]:
                continue 

            ## Unpack game data and add to dictionary
            game_data = unpack_game(next_n_lines)

            if game_data is None: 
                continue 

            game_dict[game_id] = game_data
            game_id += 1

    print('Initialized Game Dictionary')
    print('Number of Games: ', len(game_dict))

    with open(outfile_name, 'w') as fp:
        json.dump(game_dict, fp)

    return game_dict
                

def main():
    with open('config.json') as cfg_file:
        cfg = json.load(cfg_file)
    
    loaded_games = load_games(cfg)

if __name__ == '__main__':
    main()