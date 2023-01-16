############
## Python ##
############
import torch 
import pandas as pd
import numpy as np 
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import random
import argparse

#############
## Modules ## 
#############
from recommender_model import ChessGamesDataset, MatrixFactorization, get_dataloader, train_epochs, predict_user

##########
## CUDA ##
##########
if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

print('--' * 25)
print('Using ', device)
print('--' * 25)

def train_test_split(input_df, output_df, train_split=0.8):
    idx_list = list(input_df.index)
    random.shuffle(idx_list)

    split_ind = round(len(idx_list) * 0.8)

    train_ind = idx_list[:split_ind]
    test_ind = idx_list[split_ind:]

    train_input = input_df.iloc[train_ind]
    test_input = input_df.iloc[test_ind]

    train_input_np = train_input.to_numpy()
    test_input_np = test_input.to_numpy()

    train_input_ts = torch.tensor(train_input_np).to(torch.int32)
    test_input_ts = torch.tensor(test_input_np).to(torch.int32)

    train_output = output_df.iloc[train_ind]
    test_output = output_df.iloc[test_ind]

    train_output_np = train_output.to_numpy()
    test_output_np = test_output.to_numpy()

    train_output_ts = torch.tensor(train_output_np).to(torch.int32)
    test_output_ts = torch.tensor(test_output_np).to(torch.int32)

    return train_input_ts, train_output_ts, test_input_ts, test_output_ts


def prep_df(feature_df, label_df, train_split = 0.8):
    
    # Isolate useful columns 
    label_output_df = label_df.iloc[:, 4:]
    
    # Get metadata 
    feature_input_df = feature_df.iloc[:, 8:-1]
    feature_input_df.insert(0, 'user_id', label_df['user_id'])
    unique_users = len(set(feature_input_df.user_id))
    unique_openings = max(feature_df.opening_simple_id) + 1

    # Train test split
    train_input_ts, train_output_ts, test_input_ts, test_output_ts = train_test_split(feature_input_df, label_output_df, train_split=train_split)

    # Additional metadata 
    num_input_features = train_input_ts.size(dim=1) - 1

    return feature_input_df, (unique_users, unique_openings, num_input_features), (train_input_ts, train_output_ts, test_input_ts, test_output_ts)


def train_test_pipeline(metadata_tuple, df_tuple, batch_size = 64, emb_size = 500, epochs = 10):

    num_users, num_openings, num_input_features = metadata_tuple
    train_input_ts, train_output_ts, test_input_ts, test_output_ts = df_tuple

    games_dataset = ChessGamesDataset(train_input_ts, train_output_ts)
    dataloader = get_dataloader(games_dataset, batch_size = batch_size)

    test_games_dataset = ChessGamesDataset(test_input_ts, test_output_ts)
    test_dataloader = get_dataloader(test_games_dataset, batch_size = 1024)

    model = MatrixFactorization(num_users, num_input_features, num_openings, emb_size)
    train_epochs(dataloader, test_dataloader, model, epochs=epochs, lr=0.001, wd=0.0)

    return model 

def main(args):

    epochs = args.epochs
    batch_size = args.batch_size
    write = args.write

    # Load data
    feature_df = pd.read_csv('data/feature_df.csv')
    label_df = pd.read_csv('data/label_df.csv')
    id_to_username_dict = dictionary = dict(zip(feature_df.user_id, feature_df.user_name))
    username_to_id_dict = dict(zip(feature_df.user_name, feature_df.user_id))
    print('——' * 25)
    print('Read in Data')
    print('——' * 25)

    # Prep data
    feature_input_df, metadata_tuple, df_tuple = prep_df(feature_df, label_df)
    print('——' * 25)
    print('Prepped Data')
    print('——' * 25)

    # Train model
    trained_model = train_test_pipeline(metadata_tuple, df_tuple, batch_size = batch_size, epochs = epochs)
    print('——' * 25)
    print('Trained Model')
    print('——' * 25)

    # Save model
    if write == 'Y':
        torch.save(trained_model.state_dict(), 'model.pth')
        print('——' * 25)
        print('Saved Model State')
        print('——' * 25)

    # Sample Prediction 
    print('——' * 25)
    print('Sample Prediction')
    user_id = random.choice(feature_df.user_id)
    predict_user(feature_df, label_df, user_id, trained_model, id_to_username_dict)
    print('——' * 25)
    print('')

    print('——' * 25)
    print('Completed Full Pipeline')
    print('——' * 25)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--write", "-w", type=str, default='Y')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)