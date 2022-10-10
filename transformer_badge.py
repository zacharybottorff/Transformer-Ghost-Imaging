import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models. Derived class from torch.nn.Module base class.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        Default constructor for EncoderDecoder.
        """
        # Create object of superclass torch.nn.Module
        # Syntax compatible with Python 2 but not preferred for Python 3
        super(EncoderDecoder, self).__init__()
        # set EncoderDecoder.encoder to be argument encoder
        self.encoder = encoder
        # set EncoderDecoder.decoder to be argument encoder
        self.decoder = decoder
        # set EncoderDecoder.src_embed to be argument src_embed
        self.src_embed = src_embed
        # set EncoderDecoder.tgt_embed to be argument tgt_embed
        self.tgt_embed = tgt_embed
        # set EncoderDecoder.generator to be argument generator
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Take in and process masked src and target sequences.
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
        Use encoder data attribute with given src and src_mask.
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        Use decoder data attribute with given memory, src_mask, tgt, and tgt_mask.
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    """
    Define standard linear + softmax generation step. Derived class from torch.nn.Module base class.
    """
    def __init__(self, d_model, vocab):
        """
        Default constructor for Generator.
        """
        # Create object of superclass torch.nn.Module
        # Syntax compatible with Python 2 but not preferred for Python 
        super(Generator, self).__init__()
        #
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """
        Process tensor x one step.
        """
        # Perform softmax followed by log on tensor proj(x) in dimension -1
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    """
    Produce N identical layers.
    """
    # Create a torch.nn.ModuleList that contains N deep copies of module argument
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers. Derived class from torch.nn.Module base class.
    """
    # Default constructor
    def __init__(self, layer, N):
        # Create object of superclass nn.Module
        super(Encoder, self).__init__()
        # set Encoder.layers to be a torch.nn.ModuleList of N clones of argument layer
        self.layers = clones(layer, N)
        # set Encoder.norm to be the LayerNorm with features of layer.size
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Pass the input tensor (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    """
    Construct a layernorm module, see https://arxiv.org/abs/1607.06450 for details.
    """
    # Default constructor
    def __init__(self, features, eps=1e-6):
        # Create object of superclass nn.Module
        super(LayerNorm, self).__init__()
        # Set LayerNorm.a_2 to be a tensor of ones with size given by argument features
        self.a_2 = nn.Parameter(torch.ones(features))
        # Set LayerNorm.b_2 to be a tensor of zeroes with size given by argument features
        self.b_2 = nn.Parameter(torch.zeros(features))
        # Set LayerNorml.eps to be argument eps
        self.eps = eps

    def forward(self, x):
        """
        Normalize the tensor.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    # Default constructor
    def __init__(self, size, dropout):
        # Create object of superclass torch.nn.Module
        super(SublayerConnection, self).__init__()
        # Set SublayerConnection.norm to be the LayerNorm of a given size
        self.norm = LayerNorm(size)
        # Set SublayerConnection.dropout to be torch.nn.Dropout with given dropout input tensor
        # Dropout randomly sets some values to zero with a Bernoulli distribution to prevent co-adaptation of neurons 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward.
    """
    # Default constructor
    def __init__(self, size, self_attn, feed_forward, dropout):
        # Create object of superclass nn.Module
        super(EncoderLayer, self).__init__()
        # Set EncoderLayer.self_attn to be argument self_attn
        self.self_attn = self_attn
        # Set EncoderLayer.feed_forward to be argument feed_forward
        self.feed_forward = feed_forward
        # Set EncoderLayer.sublayer to be a torch.nn.ModuleList of 2 clones of a SublayerConnection of given size and dropout
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # Set EncoderLayer.size to be argument size
        self.size = size

    def forward(self, x, mask):
        """
        Apply EncoderLayer to an input tensor with a given mask.
        """
        # Change tensor x to be row 0 of EncoderLayer.sublayer
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # Return row 1 of EncoderLayer.sublayer using input tensor and Encoderlayer.feedforward
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """
    # Default constructor
    def __init__(self, layer, N):
        # Create object of superclass torch.nn.Module
        super(Decoder, self).__init__()
        # Set Decoder.layers to be a torch.nn.ModuleList of argument N clones of argument layer
        self.layers = clones(layer, N)
        # Set Decoder.norm to be the LayerNorm of size layer.size
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Pass input tensor (and masks) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward.
    """
    # Default constructor
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        # Create object of superclass torch.nn.Module
        super(DecoderLayer, self).__init__()
        # Set DecoderLayer.size to be argument size
        self.size = size
        # Set DecoderLayer.self_attn to be argument self_attn
        self.self_attn = self_attn
        # Set DecoderLayer.src_attn to be argument src_attn
        self.src_attn = src_attn
        # Set DecoderLayer.feed_forward to be argument feed_forward
        self.feed_forward = feed_forward
        # Set DecoderLayer.sublayer to be a torch.nn.ModuleList of 3 clones of a SublayerConnection of given size and dropout
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Pass the input tensor through each layer to decode with attention.
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    """
    Mask out subsequent positions. This prevents training from accessing later information.
    """
    # Create a tuple with 1 and size x size
    attn_shape = (1, size, size)
    # Creates a numpy array of ones and zeroes, with the upper triangle being 1 and the rest being 0
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # Convert array to tensor
    # Return bool whether subsequent mask is all zeroes
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    """
    #
    d_k = query.size(-1)
    # Create tensor of scores ...
    # consisting of matrix product of argument query and ...
    # the transpose around the last two dimensions of key ...
    # divided by the square root of d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # If there is mask in function call
    if mask is not None:
        # Apply the mask, filling in 0 wherever mask is 1
        # This is performed out-of-place, where masked_fill_() would perform in-place
        scores = scores.masked_fill(mask == 0, -1e9)
    # Create a tensor applying softmax to the last dimension of scores tensor
    p_attn = F.softmax(scores, dim=-1)
    # If there is dropout value in function call
    if dropout is not None:
        # Apply dropout to p_attn
        p_attn = dropout(p_attn)
    # Return tuple containing the matrix product of p_attn and argument value, p_attn
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    Implements 'Multi-Head Attention' proposed in the paper.
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # lut => lookup table
        self.lut = nn.Embedding(vocab, d_model)
        # vocab = 62 d_model = 512
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def make_model(src_vocab, tgt_vocab, N=12, d_model=1024, d_ff=2048, h=8, dropout=0.1):
    """
    Helper: Construct a model from hyperparameters.
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class SimpleLossCompute(object):
    """
    A simple loss compute and train function.
    """

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        # x = [2,3135,512]
        # y = [2,3135]
        x = self.generator(x)
        # x = [2,3135,16]
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm