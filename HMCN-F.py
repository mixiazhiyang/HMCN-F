import torch
import torch.nn as nn
import numpy as np


class Dense(nn.Module):
    def __init__(self,label_size, activation):
        super().__init__()
        self.label_size=label_size
        self.activation=activation
        self.linear=None

    def forward(self,x):
        if not isinstance(self.linear,nn.Module):
            self.linear=nn.Linear(x.shape[-1],self.label_size)
        x=self.linear(x)
        x=self.activation(x)
        return x


def local_model(num_labels, dropout_rate, relu_size):
    model = nn.Sequential(
        Dense(relu_size, activation=nn.ReLU()),
        nn.Dropout(dropout_rate),
        Dense(num_labels, activation=nn.Sigmoid())
    )
    return model


def global_model(dropout_rate, relu_size):
    model = nn.Sequential(
        Dense(relu_size, activation=nn.ReLU()),
        nn.Dropout(dropout_rate)
    )
    return model


def sigmoid_model(label_size):
    model = nn.Sequential(Dense(label_size, activation=nn.Sigmoid()))
    return model


class HMCNFModel(nn.Module):
    def __init__(self,features_size, label_size, hierarchy, beta=0.5, dropout_rate=0.1, relu_size=384):
        super().__init__()
        self.features_size=features_size
        self.label_size=label_size
        self.hierarchy=hierarchy
        self.beta=beta
        self.dropout_rate=dropout_rate
        self.relu_size=relu_size

        self.sigmoid_model=sigmoid_model(label_size)
        self.global_models=nn.ModuleList([global_model(self.dropout_rate, self.relu_size) for i in range(len(hierarchy))])
        self.local_models=nn.ModuleList([local_model(hierarchy[i], self.dropout_rate, self.relu_size) for i in range(len(hierarchy))])

    def forward(self,x):
        global_models = []
        local_models = []

        for i in range(len(hierarchy)):
            if i == 0:
                global_models.append(self.global_models[i](x))
            else:
                global_models.append(self.global_models[i](torch.cat([global_models[i - 1], x],dim=1)))
        p_glob = self.sigmoid_model(global_models[-1])

        for i in range(len(hierarchy)):
            local_models.append(self.local_models[i](global_models[i]))

        p_loc = torch.cat(local_models,dim=1)
        labels=(1-self.beta) * p_glob+self.beta * p_loc
        return labels


if __name__=='__main__':
    # linear=Dense(2,activation=nn.Sigmoid())
    # x=torch.zeros([10,5])
    # y=linear(x)
    # print(y.shape)
    # y=linear(x)
    # print(y.shape)
    # torch.save(linear,'linear.pt')
    # model=torch.load('linear.pt')
    # y=model(x)
    # print(y.shape)

    x=torch.cat([torch.zeros([1000,77]),torch.ones([628,77])],dim=0)
    y=torch.cat([torch.zeros([1000,499]),torch.ones([628,499])],dim=0)

    # hierarchy-sizes [18, 80, 178, 142, 77, 4, 0, 0], sum=499
    # (1628, 77)
    # (1628, 499

    hierarchy = [18, 80, 178, 142, 77, 4]
    feature_size = x.shape[1]
    label_size = y.shape[1]
    beta = 0.5
    model = HMCNFModel(features_size=77, label_size=499, hierarchy=hierarchy, beta=0.5, dropout_rate=0.1, relu_size=384)
    y_pred = model(x)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    for t in range(100):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)
        # print(y,y_pred)
        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 10 == 9:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    y_predict = torch.where(y_pred > 0.5, 1, 0)
    y = y.numpy()
    y_predict = y_predict.numpy()
    predict_ok = np.where(np.sum(y_predict - y, axis=1) == 0, 1, 0)
    print("{} good out of {} samples".format(np.sum(predict_ok), predict_ok.shape[0]))
