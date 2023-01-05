# Recommending Chess Openings based on Playing Style 

Authors: Adi Srikanth and Will Gorick 

Time: Winter 2023

Tags: Chess, Recommender Systems

# Overview 

## General Goal

This general goal of this project is to, for any active lichess user: 
1. Understand their style of play 
2. Approximate their style of play to that of other lichess users 
3. Use play style similarity information to power a recommender system that recommends what opening a user should play based on 
    - What players with similar playing styles play 
    - What players with similar playing styles play AND have success with 
4. Use play style similarity information to assign every user a "Celeb GM/IM" they are most similar to.  

## Tasks 

- [x] Develop overal goal/plan
- [ ] Formalize data pipeline 
- [ ] Data storage 
- [ ] Feature Engineering 
- [ ] Similarity Scoring 
- [ ] Recommender System 
- [ ] Front End/Serving 

# Setup

## Environment 

You can install the required packages for this project using the command `pip install -r requirements.txt`

We use Python version >= 3.8

## Repository Structure 

```
[ROOT]
│
└───data 
│   |   lichess_2019_06.pgn (_must upload yourself_)
│   |   lichess_2015_06.pgn (_must upload yourself_)
│   |   lichess_2013_06.pgn (_must upload yourself_)
│   |   processed_2019_06.json (_generated after running process_data.py_)
│   |   processed_2019_06_df.csv (_generated after running open_data.py_)
│
└───src
│   |   open_data.py
│   |   process_data.py
│
│
└───gitignore
│
└───config.json
│
└───requirements.txt
│
└───README.md
```

# Data Processing 

## Ingesting Raw Data 

We source our data from the lichess open [database](https://database.lichess.org/). Specifically, we download our data in the `.pgn.zst` format and extract it as one `.pgn` file. 

Next, we run our data through the `process_data.py` script. This handles the following: 
- Parses the single file from the lichess database and extracts individual games 
- Extracts the user, user elo, opening information, and PGN data for each game 
- Stores each game in a dictionary 

## Further Processing and Wrangling 

We additionally include the script `open_data.py`, which can "open up" our data and store it in the form of a pandas dataframe. This is useful for doing basic exploratory data analysis on our dataset. Additionally, we are able to easily confirm the number of unique players we have in our dataset. 