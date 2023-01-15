import torch 
import pandas as pd 

class MatrixFactorization(torch.nn.Module):
    '''
    Matrix Factorization Model 

    The users are individual players, denoted by their user ids 
    The items are individual openings, denoted by their simple opening ids 
    '''

    def __init__(self, num_users, num_items, emb_size=100):
        super(MatrixFactorization, self).__init__()

        self.user_emb = torch.nn.Embedding(num_users, emb_size)
        self.item_emb = torch.nn.Embedding(num_items, emb_size)

        # initializing our matrices with a positive number generally will yield better results
        self.user_emb.weight.data.uniform_(0, 0.5)
        self.item_emb.weight.data.uniform_(0, 0.5)

    def forward(self, u, v):
        u = self.user_emb(u)
        v = self.item_emb(v)

        return (u*v).sum(1)  # taking the dot product

def train_epocs(model, epochs=10, lr=0.01, wd=0.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    
    for i in range(epochs):
        usernames = torch.LongTensor(train_df.UserId.values)
        game_titles = torch.LongTensor(train_df.TitleId.values)
        ratings = torch.FloatTensor(train_df.Userscore.values)
        y_hat = model(usernames, game_titles)
        loss = F.mse_loss(y_hat, ratings)
        optimizer.zero_grad()  # reset gradient
        loss.backward()
        optimizer.step()
        print(loss.item())
    test(model)
