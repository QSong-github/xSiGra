import torch
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm, TransformerConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap


# Autoencoder model
class TransImg(torch.nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()

        self.final_conv_grads = None

        [in_dim, img_dim, num_hidden, out_dim] = hidden_dims

        self.conv1 = TransformerConv(in_dim, num_hidden)
        self.conv2 = TransformerConv(num_hidden, out_dim)
        self.conv3 = TransformerConv(out_dim, num_hidden)
        self.conv4 = TransformerConv(num_hidden, in_dim)

        self.imgconv1 = TransformerConv(img_dim, num_hidden)
        self.imgconv2 = TransformerConv(num_hidden, out_dim)
        self.imgconv3 = TransformerConv(out_dim, num_hidden)
        self.imgconv4 = TransformerConv(num_hidden, in_dim)

        self.neck = TransformerConv(out_dim * 2, out_dim)
        self.neck2 = TransformerConv(out_dim, out_dim)
        self.c3 = TransformerConv(out_dim, num_hidden)
        self.c4 = TransformerConv(num_hidden, in_dim)

        self.norm1 = LayerNorm(num_hidden)
        self.norm2 = LayerNorm(out_dim)

        self.activate = F.elu

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, features, img_feat, edge_index, batch_index):
        h1 = self.conv1(features, edge_index)
        if features.requires_grad == True:
            features.retain_grad()
            features.register_hook(self.activations_hook)
        h1a = self.activate(h1)
        h2 = self.conv2(h1a, edge_index)
        h3 = self.activate(self.conv3(h2, edge_index))
        h4 = self.conv4(h3, edge_index)

        img1 = self.activate(self.imgconv1(img_feat, edge_index))
        img2 = self.imgconv2(img1, edge_index)
        img3 = self.activate(self.imgconv3(img2, edge_index))
        img4 = self.imgconv4(img3, edge_index)

        concat = torch.cat([h2, img2], dim=1)
        combine = self.activate(self.neck(concat, edge_index))
        c2 = self.neck2(combine, edge_index)
        c3 = self.activate(self.c3(c2, edge_index))
        c4 = self.c4(c3, edge_index)
        hidden = gap(c2, batch_index)
        return h2, img2, c2, h4, img4, c4, hidden

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.final_conv_grads

    # method for the activation exctraction
    def get_activations(self, x, edge_index):
        return x


# Layers to convert leiden clusters to probability
class ClusteringLayer(torch.nn.Module):
    def __init__(self, num_clusters):
        super().__init__()
        self.out1 = torch.nn.Linear(30, 1024)
        self.out2 = torch.nn.Linear(1024, num_clusters)
        self.activate = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, features):
        hidden = features
        out = self.dropout(self.activate(self.out1(hidden)))
        out = self.out2(out)
        return out


# Graph autoencoder with additional probability layer
class xSiGraModel(torch.nn.Module):
    def __init__(self, hidden_dims, pretrained_model, num_clusters):
        super().__init__()
        [in_dim, img_dim, num_hidden, out_dim] = hidden_dims

        self.pretrained_model = pretrained_model
        self.out1 = torch.nn.Linear(30, 1024)
        self.out2 = torch.nn.Linear(1024, num_clusters)

        self.activate = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.1)

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, features, img_feat, edge_index, batch_index):
        h2, img2, final_conv_acts, h4, img4, c4, gap = self.pretrained_model.forward(features, img_feat, edge_index, batch_index)
        out = self.dropout(self.activate(self.out1(final_conv_acts)))
        out = self.out2(out)
        return out
