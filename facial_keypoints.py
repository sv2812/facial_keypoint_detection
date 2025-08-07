import sys
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pytorch_lightning as pl
from tqdm import tqdm as tqdm_notebook

import torch
import torch.nn.functional as F


df_training_csv = pd.read_csv('training.csv')
df_test_csv = pd.read_csv('test.csv')

print(df_training_csv.shape, df_test_csv.shape)

df_training_csv["Image_Numpy"] = [ np.array(x.split(" ")).reshape(96, 96).astype("float") for x in df_training_csv["Image"]]
df_test_csv["Image_Numpy"] = [ np.array(x.split(" ")).reshape(96, 96).astype("float") for x in df_test_csv["Image"]]

num_samples = 4
ids = np.random.randint(0, len(df_training_csv), num_samples)
for i, img in enumerate(df_training_csv.loc[ids, "Image_Numpy"].values):
    plt.subplot(1, num_samples, i+1)
    plt.imshow(img, cmap="gray")
plt.show()

print("Total training examples:", len(df_training_csv))

print("Column wise null values:")
df_training_csv.isna().sum()

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Impute train.
df_temp = df_training_csv.drop(["Image","Image_Numpy"], axis=1)
df_temp = pd.DataFrame(imputer.fit_transform(df_temp), columns=list(df_temp.columns))
df_temp["Image"] = df_training_csv["Image"].values
df_temp["Image_Numpy"] = df_training_csv["Image_Numpy"].values
df_training_csv = df_temp.reset_index(drop=True)

out_col_names = list(df_training_csv.drop(["Image", "Image_Numpy"], axis=1).columns)
in_col_names = "Image_Numpy"

keypoint_feature_prefixes = [
 'left_eye_center',
 'left_eye_inner_corner',
 'left_eye_outer_corner',
 'left_eyebrow_inner_end',
 'left_eyebrow_outer_end',
 'mouth_center_bottom_lip',
 'mouth_center_top_lip',
 'mouth_left_corner',
 'mouth_right_corner',
 'nose_tip',
 'right_eye_center',
 'right_eye_inner_corner',
 'right_eye_outer_corner',
 'right_eyebrow_inner_end',
 'right_eyebrow_outer_end']

def plot_image_with_points(img, keypoints):
#     ncols = 4
#     keypoints = np.squeeze(keypoints)
#     if len(keypoints.shape)==1:
#         keypoints = np.expand_dims(keypoints, axis=0)
    keypoints = keypoints.reshape(len(keypoints)//2,2)
    plt.plot(np.squeeze(keypoints[:,0]), np.squeeze(keypoints[:,1]), "or")
    plt.imshow(img, cmap="gray")
    plt.show()

idx = int(np.random.randint(0, len(df_training_csv), 1))

img = df_training_csv.loc[idx, in_col_names]
print("img", img.shape)
keypoints = df_training_csv.loc[idx, out_col_names].values
plot_image_with_points(img, keypoints)

df_train, df_val = train_test_split(df_training_csv.sample(frac=1).reset_index(drop=True), train_size=0.8)

scaler = StandardScaler()
scaler.fit(df_train[out_col_names])

df_train[out_col_names] = scaler.transform(df_train[out_col_names])
df_val[out_col_names] = scaler.transform(df_val[out_col_names])

df_train.shape, df_val.shape


x_train = torch.Tensor(np.array(list(df_train[in_col_names].values)).astype("float")/255.0)
x_train = torch.unsqueeze(x_train, axis=1)
y_train = torch.Tensor(df_train[out_col_names].values.astype("float"))


x_val = torch.Tensor(np.array(list(df_val[in_col_names].values)).astype("float")/255.0)
x_val = torch.unsqueeze(x_val, axis=1)
y_val = torch.Tensor(df_val[out_col_names].values.astype("float"))

ds_train = torch.utils.data.TensorDataset(x_train[:], y_train[:])
ds_val = torch.utils.data.TensorDataset(x_val[:], y_val[:])

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=16)
dl_val = torch.utils.data.DataLoader(ds_val, batch_size=16)

class ModelCNN(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=30):
        super(ModelCNN, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1),
            torch.nn.Dropout(0.3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1),
            torch.nn.Dropout(0.3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1),
            torch.nn.Dropout(0.3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1),
            torch.nn.Dropout(0.3),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Flatten(),

            torch.nn.Linear(1024, 512),
            torch.nn.Dropout(0.3),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),

            torch.nn.Linear(512, out_channels)

        )

    def forward(self, x):
        x = self.model(x)
        return x


class Learner(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    # Create the model and learner
    model = ModelCNN(in_channels=1, out_channels=30)
    learner = Learner(model, lr=1e-3)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=6,
        accelerator='auto',  # Will use GPU if available
        devices='auto',
        log_every_n_steps=10,
    )

    # Train the model
    trainer.fit(learner, dl_train, dl_val)



