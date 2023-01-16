import torch 
from tqdm.auto import tqdm 

########################
## Model Architecture ##
########################

class ChessGamesDataset(torch.utils.data.Dataset):
    def __init__(self, input_ts, output_ts):

        self.xdata = input_ts
        self.ydata = output_ts

    def __len__(self):
        return len(self.xdata)

    def __getitem__(self, idx):
      input_row = self.xdata[idx, :]
      output_row = self.ydata[idx, :]

      return input_row, output_row

class MatrixFactorization(torch.nn.Module):
    '''
    Matrix Factorization Model 

    The users are individual players, denoted by their user ids 
    The items are individual players, denoted by their features/attributes
    '''

    def __init__(self, num_users, num_input_features, unique_openings, emb_size=400):
        super(MatrixFactorization, self).__init__()
        
        # initializing our matrices with a positive number generally will yield better results
        self.user_emb = torch.nn.Embedding(num_users, emb_size)
        self.user_emb.weight.data.uniform_(0.1, 0.75)

        # learn to represent the input features 
        self.l1 = torch.nn.Linear(num_input_features, emb_size, dtype=torch.float32)

        # activation 
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()
        self.act3 = torch.nn.ReLU()

        # produce predictions for the best opening evals (for all openings)
        self.l2 = torch.nn.Linear(emb_size, emb_size * 2)
        self.l3 = torch.nn.Linear(emb_size * 2, unique_openings * 2)
        self.l4 = torch.nn.Linear(unique_openings * 2, unique_openings)


    def forward(self, u, v):

        # layer 1
        u_out = self.user_emb(u)
        v_out = self.l1(v.to(torch.float32))
        v_out = self.act1(v_out)
        out1 = torch.mul(u_out, v_out)

        # layer 2 
        out2 = self.l2(out1)
        out2 = self.act2(out2)

        # layer 3 
        out3 = self.l3(out2)
        out3 = self.act3(out3)

        # layer 4 
        out4 = self.l4(out3)

        return out4 

#############
## Methods ##
#############

def get_dataloader(dataset, batch_size = 32):
  return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_epochs(dataloader, test_dataloader, model, epochs=10, lr=0.01, wd=0.0, device='cpu', custom_loss=False):
    
    # Setting up optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    model.to(device)
    criterion = torch.nn.MSELoss()

    for i in tqdm(range(epochs)):
      epoch_loss = 0
      for input_ts, output_ts in tqdm(dataloader):
        # Isolate user ids and user attributes
        user_ids = input_ts[:, 0].to(device)
        user_info = input_ts[:, 1:].to(device)

        # Prepare output 
        output_ts = output_ts.to(torch.float32)
        output_ts = output_ts.to(device)

        # Generate predictions and calculate loss 
        y_predictions = model(user_ids, user_info)

        loss = criterion(y_predictions, output_ts)
        epoch_loss += loss.item()

        # Step through model 
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

      print('')
      print('Average Epoch Loss : ', epoch_loss / len(dataloader))
      
      test_loss = test(model, dataloader, custom_loss = custom_loss)
      print('Test Loss After Epoch ', i + 1, ':', test_loss)
      print('')

def test(model, dataloader, custom_loss = False, device='cpu'):
    model.eval()
    criterion = torch.nn.MSELoss()

    total_loss = 0 

    with torch.no_grad(): 
      for input_ts, output_ts in dataloader:
          user_ids = input_ts[:, 0].to(device)
          user_info = input_ts[:, 1:].to(device)

          # Prepare output 
          output_ts = output_ts.to(torch.float32)
          output_ts = output_ts.to(device)

          # Generate predictions and calculate loss 
          y_predictions = model(user_ids, user_info)
          loss = criterion(y_predictions, output_ts)  

          total_loss += loss.item() 
    
    return round(total_loss / len(dataloader), 4)

def row_to_ts(row):
  row_np = row.to_numpy()
  row_ts = torch.tensor(row_np).to(torch.int32)
  return row_ts

def model_predict(trained_model, ts, device='cpu'):
  ts = ts.unsqueeze(dim=0)
  user_id = ts[:, 0].to(device)
  user_info = ts[:, 1:].to(device)  

  out = trained_model(user_id, user_info)
  return out

def top_openings(model_output, n, opening_dict):
  top_n = torch.topk(model_output.flatten(), n).indices.tolist()
  top_n_names = [opening_dict[x] for x in top_n]

  redo = 0
  for name in top_n_names:
    if 'efence' in name or 'efense' in name:
        redo += 1

  if redo > 0:
    top_100 = torch.topk(model_output.flatten(), 100).indices.tolist()
    top_100_names = [opening_dict[x] for x in top_100]
    
    top_n_names = []
    for name in top_100_names:
        if 'efence' not in name and 'efense' not in name:
            top_n_names.append(name)
            if len(top_n_names) == n:
                break

  return top_n_names 

def predict_user(feature_df, label_df, id, trained_model, id_to_username_dict = None, n = 3, device = 'cpu'):

  df = feature_df.iloc[:, 8:-1]
  df.insert(0, 'user_id', label_df['user_id'])

  user_df = df[df.user_id == id]
  user_row = user_df.mean()
  user_ts = row_to_ts(user_row)
  user_output = model_predict(trained_model, user_ts, device)
  
  opening_dict = dictionary = dict(zip(feature_df.opening_simple_id, feature_df.opening_name_simple))
  opening_recs = top_openings(user_output, n, opening_dict)

  if id_to_username_dict != None:
    print('Recommendations for ', id_to_username_dict[id])

  for num, opening in enumerate(opening_recs):
    print('# ', num + 1, ' Opening Recommendation :', opening)