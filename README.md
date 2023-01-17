# Recommending Chess Openings based on Play Style 

## _Project still in progress_ 

Authors: [Adi Srikanth](https://lichess.org/@/asrikanth) and Will Gorick 

Time: Winter 2023

Tags: Chess, Recommender Systems

# Overview 

## General Goal

This general goal of this project is to, for any active lichess user: 
1. Understand their style of play 
2. Approximate their style of play to that of other lichess users 
3. Use play style similarity information to power a recommender system that recommends what opening a user should play based on: 
    - What players with similar playing styles play 
    - What players with similar playing styles play AND have success with 
4. Use play style similarity information to assign every user a "Celeb GM/IM" they are most similar to.  

## Tasks 

### Current Status

We have a version 1 recommender system working powered by around 200K games from lichess. (Note: we pulled almost 70M games, but a very small fraction contains engine evaluation data, which we use in our recommendations). Our recommender system has a solid base of features and can reliably recommend openings to a user with game data. 

### Version 1 Checklist

- [X] Develop overal goal/plan
- [X] Formalize data pipeline 
- [ ] Data storage 
- [x] Feature Engineering 
- [x] Similarity Scoring 
- [x] Recommender System 
- [ ] Celeb GM/IM
- [ ] Front End/Serving 

# Setup

## Environment 

You can install the required packages for this project using the command `pip install -r requirements.txt`

We use Python version >= 3.8

## Quickstart

In order to run this without having to run the entire data pipeline, go to the Google Drive and upload the files `full_feature_df.csv` and `full_labels_df.csv`. With these two files, you will be able to run the recommender system. 

See the repository structure below for details on where to store the files. 

## Repository Structure 

```
[ROOT]
│
└───data 
│   |   full_features_df.csv (must upload - available in google drive)
│   |   full_labels_df.csv (must upload - available in google drive)
│   |   lichess_2019_06.pgn (*)
│   |   lichess_2015_06.pgn (*)
│   |   lichess_2013_06.pgn (*)
│   |   processed_2019_06.json (**)
│   |   processed_2019_06_df.csv (**)
│   |
│   |   * - must upload yourself, but only necessary to run full data pipeline
│   |   ** - generated during full data pipeline process
│
└───src
│   |   process_data.py
│   |   format_data.py
│   |   chess_utils.py
│
└───notebooks
│   |   scratch.ipynb
│
└───recommender
│   |   recommender_model.py
│   |   run_recommender.py
│
└───gitignore
│
└───config.json
│
└───requirements.txt
│
└───README.md
```

# Data Pipeline 

## Diagram 

[![](https://mermaid.ink/img/pako:eNqVlMtugzAQRX_FcjepRFDIO1Tqplm2mz5WgCIHjxO3YCNjktKo_14nLk2gjxRWxvfMGSNgdjiWFLCPWSK38ZoojR7noUDm4tQLAsYTECQFN1sJ9y3XaMMJuuXxGvI8ilC3e224fid4ErFMM7XfvURXlaAZWHxQs0ZHvBlYfGgkGSUa0I0UjK8cdF8IlCkZG-nCBMTNytOu_y2w-lHnMwHqPudSnIhGQT2qTjTuBHsjkyol-qcT_AbY8smx44IyN843J6WToBlWXadWmslc156FZ6VYnnY_B1rdzLxcILpQUPWJ2iq8XhAkZAlJzWAduS4T2DNmpeQL-BeT8cxBjr3rbjnVa3-YvdbpWSva653F6wX9L54Rr8n3m_rBX_Zv9LCVe9TKPW7lnrRyT8-6Q4EdnIL5ljk1k2K3rw-xXkMKIfbNkgIjRaJDHIp3g5JCy4dSxNjXqgAHF4efcc7JSpEU-4wkudkFyrVUd3b6HIbQ-wfJPoO2?type=png)](https://mermaid.live/edit#pako:eNqVlMtugzAQRX_FcjepRFDIO1Tqplm2mz5WgCIHjxO3YCNjktKo_14nLk2gjxRWxvfMGSNgdjiWFLCPWSK38ZoojR7noUDm4tQLAsYTECQFN1sJ9y3XaMMJuuXxGvI8ilC3e224fid4ErFMM7XfvURXlaAZWHxQs0ZHvBlYfGgkGSUa0I0UjK8cdF8IlCkZG-nCBMTNytOu_y2w-lHnMwHqPudSnIhGQT2qTjTuBHsjkyol-qcT_AbY8smx44IyN843J6WToBlWXadWmslc156FZ6VYnnY_B1rdzLxcILpQUPWJ2iq8XhAkZAlJzWAduS4T2DNmpeQL-BeT8cxBjr3rbjnVa3-YvdbpWSva653F6wX9L54Rr8n3m_rBX_Zv9LCVe9TKPW7lnrRyT8-6Q4EdnIL5ljk1k2K3rw-xXkMKIfbNkgIjRaJDHIp3g5JCy4dSxNjXqgAHF4efcc7JSpEU-4wkudkFyrVUd3b6HIbQ-wfJPoO2)

## Step 1: Ingesting Raw Data 

We source our data from the lichess open [database](https://database.lichess.org/). Specifically, we download our data in the `.pgn.zst` format and extract it as one `.pgn` file. 

Next, we run our data through the `process_data.py` script. This handles the following: 
- Parses the single file from the lichess database and extracts individual games 
- Extracts the user, user elo, opening information, and PGN data for each game 
- Stores each game in a dictionary 

In order to run this file, you must first update the `config.json` file to make sure the data stored under `live_run` points to your saved pgn file. Specifically, the `dataset_file` key should point to your pgn filename. The `processsed_file` key can point to any filename, but should generally follow naming conventions. 

Additionally, the `dataset` key points to the name of the dataset, which is also another nested data categority in the config file. Here, fill in the number of games (which you can find on the lichess database) and the number of lines in your file, which you can compute by running `wc -l <filename>` in your terminal. 

At this point, you can go ahead and run: `python src/process_data.py`

## Step 2: Further Processing and Wrangling 

We additionally include the script `open_data.py`, which can "open up" our data and store it in the form of a pandas dataframe. This is useful for doing basic exploratory data analysis on our dataset. Additionally, we are able to easily confirm the number of unique players we have in our dataset. 

To run this step, run `python src/format_data.py`. If you have strayed from the file naming conventions, you may have to adjust the script slightly. 

## Step 3: Assorted Feature Extraction 

The ipynotebook `notebooks/scratch.ipynb` contains different cells that help finish up the data pipeline stage. These steps need to be ported over to Python, but this has been deprioritized for the time being. 

The cells of importance are the cells that generated the Y Data (Label Data) and the cells that append data (this is helpful to add more data to our current store). 

## Utils

The file `chess_utils.py` contains various utility functions and is called on throughout the data pipeline process. 

The file has methods for both individual pieces of data, but also has methods that apply those same transformations to an entire dataframe. 

# Recommender System 

## Overview 

The recommender system used is fairly simple, and rests on two main components: 
1. Embeddings 
2. Feed-Forward Network 

### Embeddings 

The embeddings are standard Pytorch embeddings. Specifically, they take a single user id and represents the user id as a n-dimensional vector. This is useful because the embeddings can be learned and fine tuned for a single user (after multiple rounds of learning) and the embeddings can store information specific to the user. 

### Feed Forward Network 

The feed-forward network uses three linear layers and ReLU activation functions. The first linear layer takes the user attribute data and runs it through a layer. The next two layers are used to expand dimensionality and ultimately produce an output vector that is of size p, where p is the number of unique openings we are considering. Ultimately, the goal is for the output vector to predict the opening evaluation a user would end up with if they played a certain opening. 

## Features 

We use the following features in order to describe a user's playing style: 
- `opening_eval`: aggregate engine eval of the last 3 moves of the opening 
- `opening_id_n`: indicator variable for whether or not an opening of id n was played 
- `move 5 <piece>`: number of times each piece type has been moved in the first 5 moves
- `move 10 <piece>`: number of times each piece type has been moved between moves 5-10
- `move 15 <piece>`: number of times each piece type has been moved between moves 5-15
- `move final <piece>`: number of times each piece type has been moved between moves 15-end of game
- `move N captures`: number of captures for the four move categories noted above 
- `move N checks`: number of checks given for the four move categories noted above 
- `move N pawn density`: number of pawns in central 16 squares for the four move categories noted above 

## Files and Code 

### Architecture

The architecture of the model is created and stored in `recommender_model.py`. Here we define the following: 
- `ChessGamesDataset()`: Custom dataset
- `get_dataloader()`: dataloader 
- `MatrixFactorization()`: Model 
- `train_epochs():` Training Loop 
- `test()`: test/validation function 
- `predict_user()`: function to generate recommendations for a single user

### Execution 

The actual training of the model is conducted in `run_recommender.py`. We note here that by default, the GPU accelerator is defined to be `mps`, which supports the newest fleet of Macbooks (M1). If using a virtual machine such as an NVidia cluster, this should be changed to support `cuda`. Otherwise, the program will default to `cpu`. 

You can run the recommender by running `python recommender/run_recommender.py`. The script takes various command line arguments. To clarify this, the argument parsing function is included below: 
```
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--alpha", "-a", type=float, default=0.001)
    parser.add_argument("--write", "-w", type=str, default='Y')
    parser.add_argument("--loss", "-l", type=str, default='MSE')

    return parser.parse_args()
```

Here is a sample run that utilizes all of the command line arguments: 

`python recommender/run_recommender.py --epochs 10 --batch_size 128 --alpha 0.01 --write Y --loss MSE`

### Training Time 

Using a Python 3 Google Compute Engine backend (High-Ram and Premium GPU), we can complete 10 epochs of training in around 30 minutes. 


## Sample Prediction 

Below is a sample prediction for the lichess user [Igracsaha](https://lichess.org/@/Igracsaha). 

```
——————————————————————————————————————————————————
Sample Prediction
Recommendations for  Igracsaha
#  1  Opening Recommendation : Queen's Pawn Game
#  2  Opening Recommendation : Italian Game
#  3  Opening Recommendation : English Opening
——————————————————————————————————————————————————
```

The results here are interesting. A brief OpeningTree analysis of Igracsaha's games shows that they clearly prefer _1. e4_, playing over 95% of their openings with this beginning. However, some variants of d4 openings (the #1 recommendation) score well for this user (specifically _1. Nf3_ scores better than _1. e4_). 

Interestingly enough, the #2 recommendation is the Italian Game. This is in fact the most common opening played by Igracsaha and scores around 53% for this user. 
