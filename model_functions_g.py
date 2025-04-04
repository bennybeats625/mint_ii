import numpy as np
import torch
import torch.nn as nn
import math

# this is where the magic is happening
class TransformerEncoder(nn.Module):
    def __init__(self, ntoken, em_dim, nhead, nhid, nlayers, max_len=256, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        # the embedding layer takes in a batch of token sequences of max length (alreay padded, the padding is 0)
            # it's a matrix of batch_size x max_len
        
        # each token is mapped to an embedding vector of size em_dim (initialized randomly, then updated during training)
        
        # each token gets replaced by its corresponding embedding vector of length em_dim
            # it's now a cube of batch_size x max_len x em_dim
        self.embedding = nn.Embedding(ntoken, em_dim, padding_idx=0)
        
        # transformers don't have position or sequence data; there is no innate sense of order to the tokens
        # the positional encoder takes in the embedded cube and adds positional data to it
            # it's still a cube of batch_size x max_len x em_dim
        self.pos_encoder = PositionalEncoding(em_dim, max_len, dropout)
        
        # this defines a single transformer encoder layer 
            # the encoder layer applies the multi-head self-attention and feedforward layers
                # self-attention allows each token to attend to every other token in the sequence
                # the feedforward network refines features after attention
            # there's also residual connections and normalization for stability
        
        # the encoder layer takes in a a batch of embeddings with positional encoding
            # for one layer it's a matrix of max_len x em_dim
        encoder_layers = nn.TransformerEncoderLayer(em_dim, nhead, nhid, dropout, batch_first=True)
        
        # that was all for a single encoder layer
        # this uses a bunch of layers for a full batch
            # now it's back to a cube of batch_size x max_len x em_dim
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        
        # this is the embedding dimension
        self.em_dim = em_dim
        
        # the fully connected layer maps the output of the encoder from embedding space to token space
            # from batch_size x max_len x em_dim
            # to batch_size x max_len x ntoken 
        self.output_layer = nn.Linear(em_dim, ntoken)
        
    # this is how the tokens flow through the model
    # it takes in a matrix of batch_size x max_len 
    def forward(self, src, src_mask):
        
        # print(f"1st src shape: {src.shape}")
        
        # embed the tokens
            # the embedded values are scaled down so they have similar magnitudes to the positional encodings
        src = self.embedding(src) * math.sqrt(self.em_dim)
        
        # print(f"2nd src shape: {src.shape}")
        
        # add positional encodings
        src = self.pos_encoder(src)
        
        # print(f"3rd src shape: {src.shape}")
        
        # self-attend and feedforward with the transformer itself
        # output = self.transformer_encoder(src, mask=tgt_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.transformer_encoder(src, src_mask)
        
        # fully connect (? idk)
        output = self.output_layer(output)
        
        # each token now has a score for every other token in the dictionary
        return output
        
    # make sure the transformer isn't cheating and looking at future tokens
    # use a mask to cover the future tokens
    def subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# we need to encode position manually because the transformer doesn't do it on it's own
class PositionalEncoding(nn.Module):
    def __init__(self, em_dim, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        
        # this randomly disables some neurons to help prevent overfitting
        self.dropout = nn.Dropout(p=dropout)
        
        # initialize an empty tensor
            # it's a vector of max_len x em_dim
        # this will hold the position values for the tokens
        pe = torch.zeros(max_len, em_dim)
        
        # print(f"1st pe shape: {pe.shape}")
        
        # creates a column vector of length max_len
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # you use a bunch of sin and cos waves to define position
        # the frequency depends on the position within max length
        pe[:, 0::2] = torch.sin(position * (torch.exp(torch.arange(0, em_dim, 2).float() * (-np.log(10000.0) / em_dim))))
        
        pe[:, 1::2] = torch.cos(position * (torch.exp(torch.arange(0, em_dim, 2).float() * (-np.log(10000.0) / em_dim))))
        
        # add a dimension to store the batch
        # this way you can handle multiple batches
        pe = pe.unsqueeze(0)
        
        # print(f"2nd pe shape: {pe.shape}")
        
        # this stores the position in the model without making it a parameter
        # this makes sure the positional encodings don't change during training
        self.register_buffer('pe', pe)
        
    # the forward pass takes in the embedded tokens
        # this is the cube of batch_size x max_len x em_dim
    def forward(self, x):
        
        # the positional encodings are added to the embeddings
            # as in actual addition
        # print(f"x shape: {x.shape}, pe shape: {self.pe.shape}")
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)