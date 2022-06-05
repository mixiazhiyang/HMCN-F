import torch
import torch.nn as nn


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

    def forward(self,x):
        global_models = []
        local_models = []

        for i in range(len(hierarchy)):
            if i == 0:
                global_models.append(global_model(self.dropout_rate, self.relu_size)(x))
            else:
                global_models.append(global_model(self.dropout_rate, self.relu_size)(torch.cat([global_models[i - 1], x],dim=1)))
        p_glob = sigmoid_model(label_size)(global_models[-1])

        for i in range(len(hierarchy)):
            local_models.append(local_model(hierarchy[i], self.dropout_rate, self.relu_size)(global_models[i]))

        p_loc = torch.cat(local_models,dim=1)
        labels=(1-beta) * p_glob+beta * p_loc
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

    x=torch.zeros([1628,77])
    y=torch.zeros([1628,499])

    # hierarchy-sizes [18, 80, 178, 142, 77, 4, 0, 0], sum=499
    # (1628, 77)
    # (1628, 499

    hierarchy = [18, 80, 178, 142, 77, 4]
    feature_size = x.shape[1]
    label_size = y.shape[1]
    beta = 0.5
    model = HMCNFModel(features_size=77, label_size=499, hierarchy=hierarchy, beta=0.5, dropout_rate=0.1, relu_size=384)
    p=model(x)