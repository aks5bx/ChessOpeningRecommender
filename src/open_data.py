import pandas as pd 
import json 

def dict_to_pandas(filename):
    '''
    Takes a dictionary of games and returns a pandas dataframe
    '''
    with open(filename) as f:
        game_dict = json.load(f)
        print('Loaded JSON')

        df = pd.DataFrame.from_dict(game_dict, orient='index')
        print('Converted to Pandas DataFrame')
    
    return df

def explore_df(df):
    print('')
    print('Dataframe Head')
    print(df.head())
    print('')
    print('Dataframe Info')
    print('Num Rows :', df.shape[0])
    print('Num Unique Users :', len(set(df['user_name'])))

    df_grouped = df.groupby(['user_name'])['user_name'].count().to_frame() # .to_frame().reset_index()
    df_grouped.columns = ['num_games']
    df_grouped = df_grouped.sort_values(by='num_games', ascending=False)
    print(df_grouped.head(20))
    print('')
    print('Grouped_DF')

def main():
    with open('config.json') as cfg_file:
        cfg = json.load(cfg_file)

    filename = cfg['live_run']['processsed_file'] 
    games_df = dict_to_pandas(filename)

    explore_df(games_df)

if __name__ == '__main__':
    main()