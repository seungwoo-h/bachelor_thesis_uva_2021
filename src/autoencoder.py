import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

# Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(Autoencoder, self).__init__()
        # encoder
        self.encode = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, z_dim),
        )
        
        # decoder
        self.decode = nn.Sequential(
            nn.BatchNorm1d(z_dim),
            nn.Linear(z_dim, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        x = self.encode(x)
        x_hat = self.decode(x)
        return x_hat

# Autoencoder Dataset
class AutoencoderDataset(Dataset):
    def __init__(self, x_data):
        self.data = x_data
        
    def __getitem__(self, idx):
        data = self.data[idx]
        return data
    
    def __len__(self):
        return len(self.data)

# Autoencoder Trainer
class AutoencoderTrainer:
    def __init__(self, X_unselected, args):
        X_unselected_train, X_unselected_valid = train_test_split(X_unselected, test_size=args.ae_val_size, random_state=args.seed)
        self.scaler = MinMaxScaler()
        X_unselected_train = self.scaler.fit_transform(X_unselected_train)
        X_unselected_valid = self.scaler.transform(X_unselected_valid)

        train_dataset = AutoencoderDataset(X_unselected_train)
        self.train_loader = DataLoader(train_dataset, batch_size=args.ae_batch_size, shuffle=True, num_workers=args.ae_num_workers)
        valid_dataset = AutoencoderDataset(X_unselected_valid)
        self.valid_loader = DataLoader(valid_dataset, batch_size=args.ae_batch_size, shuffle=False, num_workers=args.ae_num_workers)

        device = args.device

        self.ae_model = Autoencoder(input_dim=X_unselected_train.shape[1], z_dim=args.ae_hidden_dim)
        self.ae_model.to(device)

        self.criterion = nn.MSELoss().to(device)
    
    def load_model(self, args):
        self.ae_model.load_state_dict(torch.load(args.ae_load_model_dir))
        self.ae_model.eval()

    def train(self, args):
        print(f"Start Training - {args.ae_num_epochs} epochs.")
        loss_func = self.criterion
        device = args.device
        optimizer = torch.optim.Adam(self.ae_model.parameters(), lr=args.ae_learning_rate)
        train_loss_arr = []
        val_loss_arr = []
        for epoch in range(0, args.ae_num_epochs):
            epoch_loss = []
            epoch_val_loss = []
            for x in self.train_loader:
                self.ae_model.train()
                optimizer.zero_grad()
                
                x = x.float().to(device)
                
                output = self.ae_model.forward(x)
                loss = loss_func(output, x)
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())
            
            # Validate
            self.ae_model.eval()
            with torch.no_grad():
                for x in self.valid_loader:
                    x = x.float().to(device)
                    output = self.ae_model.forward(x)
                    val_loss = loss_func(output, x)
                    epoch_val_loss.append(val_loss.item())
            
            # Save log
            train_loss_arr.append(np.mean(epoch_loss))
            val_loss_arr.append(np.mean(epoch_val_loss))
            
            # Save model
            torch.save(self.ae_model.state_dict(), f"{args.ae_checkpoint_dir}/model-{epoch}.bin")

            if epoch % 10 == 0:
                print(f'[Epoch {epoch}] Train Loss: {round(np.mean(epoch_loss), 4)} | Val Loss: {round(np.mean(epoch_val_loss), 4)}')

        history = (train_loss_arr, val_loss_arr)
        print(f"Best checkpoint - Epoch: {np.argmin(val_loss_arr)}")
        return history
                

def _encode(loader, encoder_model):
  outputs = []
  encoder_model.eval()
  with torch.no_grad():
    for batch in loader:
      x = batch.float().to(device)
      output = encoder_model.encode(x)
      outputs.append(output)
    outputs = torch.cat(outputs).cpu().numpy()
  return outputs

