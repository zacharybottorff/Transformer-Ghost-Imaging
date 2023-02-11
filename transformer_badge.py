import logging

# Create a logger
mainlogger = logging.getLogger("mainlogger")

# Change to logging.WARNING to disable logging statements
debug_level = logging.DEBUG
mainlogger.setLevel(debug_level)

# Create file handler
logfile = "output.log"
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(debug_level)

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add handlers to logger
mainlogger.addHandler(fh)
mainlogger.addHandler(ch)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models. Derived class from torch.nn.Module base class.
    """
    # Default constructor for EncoderDecoder
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        # Create object of superclass torch.nn.Module
        # Syntax compatible with Python 2 but not preferred for Python 3
        super(EncoderDecoder, self).__init__()
        # Set EncoderDecoder.encoder to be parameter encoder
        self.encoder = encoder
        # Set EncoderDecoder.decoder to be parameter encoder
        self.decoder = decoder
        # Set EncoderDecoder.src_embed to be parameter src_embed
        self.src_embed = src_embed
        # Set EncoderDecoder.tgt_embed to be parameter tgt_embed
        self.tgt_embed = tgt_embed
        # Set EncoderDecoder.generator to be parameter generator
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Take in and process masked src and target sequences.
        """
        mainlogger.info("Calling EncoderDecoder.forward()")
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
        Use encoder data attribute with given src and src_mask.
        """
        mainlogger.info("Calling EncoderDecoder.encode()")
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        Use decoder data attribute with given memory, src_mask, tgt, and tgt_mask.
        """
        mainlogger.info("Calling EncoderDecoder.decode()")
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    """
    Define standard linear + softmax generation step. Derived class from torch.nn.Module base class.
    """
    # Default constructor for Generator
    def __init__(self, d_model, vocab):
        # Create object of superclass torch.nn.Module
        # Syntax compatible with Python 2 but not preferred for Python 3
        super(Generator, self).__init__()
        # Set proj to be linear transformation Tensor with dimensions d_model and vocab 
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """
        Process tensor x one step.
        """
        mainlogger.info("Calling Generator.forward()")
        # Perform softmax followed by log on tensor proj(x) in dimension -1
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    """
    Produce N identical layers.
    """
    mainlogger.info("Calling clones()")
    # Create a torch.nn.ModuleList that contains N deep copies of module parameter
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers. Derived class from torch.nn.Module base class.
    """
    # Default constructor for Encoder
    def __init__(self, layer, N):
        # Create object of superclass nn.Module
        super(Encoder, self).__init__()
        # set Encoder.layers to be a torch.nn.ModuleList of N clones of parameter layer
        self.layers = clones(layer, N)
        # set Encoder.norm to be the LayerNorm with features of layer.size
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Pass the input tensor (and mask) through each layer in turn.
        """
        mainlogger.info("Calling Encoder.forward()")
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    """
    Construct a layernorm module, see https://arxiv.org/abs/1607.06450 for details.
    """
    # Default constructor for LayerNorm
    def __init__(self, features, eps=1e-6):
        # Create object of superclass nn.Module
        super(LayerNorm, self).__init__()
        # Set LayerNorm.a_2 to be a tensor of ones with size given by parameter features
        self.a_2 = nn.Parameter(torch.ones(features))
        # Set LayerNorm.b_2 to be a tensor of zeroes with size given by parameter features
        self.b_2 = nn.Parameter(torch.zeros(features))
        # Set LayerNorml.eps to be parameter eps
        self.eps = eps

    def forward(self, x):
        """
        Normalize the tensor.
        """
        mainlogger.info("Calling LayerNorm.forward()")
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    # Default constructor for SublayerConnection
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
        mainlogger.info("Calling SublayerConnection.forward()")
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward.
    """
    # Default constructor for EncoderLayer
    def __init__(self, size, self_attn, feed_forward, dropout):
        # Create object of superclass nn.Module
        super(EncoderLayer, self).__init__()
        # Set EncoderLayer.self_attn to be parameter self_attn
        self.self_attn = self_attn
        # Set EncoderLayer.feed_forward to be parameter feed_forward
        self.feed_forward = feed_forward
        # Set EncoderLayer.sublayer to be a torch.nn.ModuleList of 2 clones of a SublayerConnection of given size and dropout
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # Set EncoderLayer.size to be parameter size
        self.size = size

    def forward(self, x, mask):
        """
        Apply EncoderLayer to an input tensor with a given mask.
        """
        mainlogger.info("Calling EncoderLayer.forward()")
        # Change tensor x to be row 0 of EncoderLayer.sublayer
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # Return row 1 of EncoderLayer.sublayer using input tensor and Encoderlayer.feedforward
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """
    # Default constructor for Decoder
    def __init__(self, layer, N):
        # Create object of superclass torch.nn.Module
        super(Decoder, self).__init__()
        # Set Decoder.layers to be a torch.nn.ModuleList of parameter N clones of parameter layer
        self.layers = clones(layer, N)
        # Set Decoder.norm to be the LayerNorm of size layer.size
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Pass input tensor (and masks) through each layer in turn.
        """
        mainlogger.info("Calling Decoder.forward()")
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward.
    """
    # Default constructor for DecoderLayer
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        # Create object of superclass torch.nn.Module
        super(DecoderLayer, self).__init__()
        # Set DecoderLayer.size to be parameter size
        self.size = size
        # Set DecoderLayer.self_attn to be parameter self_attn
        self.self_attn = self_attn
        # Set DecoderLayer.src_attn to be parameter src_attn
        self.src_attn = src_attn
        # Set DecoderLayer.feed_forward to be parameter feed_forward
        self.feed_forward = feed_forward
        # Set DecoderLayer.sublayer to be a torch.nn.ModuleList of 3 clones of a SublayerConnection of given size and dropout
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Pass the input tensor through each layer to decode with attention.
        """
        mainlogger.info("Calling DecoderLayer.forward()")
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    """
    Mask out subsequent positions. This prevents training from accessing later information.
    """
    mainlogger.info("Calling subsequent_mask()")
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
    mainlogger.info("Calling attention()")
    # Set d_k to be size of last dimension of query
    d_k = query.size(-1)
    mainlogger.debug("d_k = %s", d_k)
    # Create tensor of scores ...
    # consisting of matrix product of parameter query and ...
    # the transpose around the last two dimensions of key ...
    # divided by the square root of d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    mainlogger.debug("scores = %s", scores)
    # If there is mask in function call
    if mask is not None:
        # Apply the mask, filling in 0 wherever mask is 1
        # This is performed out-of-place, where masked_fill_() would perform in-place
        scores = scores.masked_fill(mask == 0, -1e9)
        mainlogger.debug("scores = %s", scores)
    # Create a tensor applying softmax to the last dimension of scores tensor
    p_attn = F.softmax(scores, dim=-1)
    mainlogger.debug("p_attn = %s", p_attn)
    # If there is dropout value in function call
    if dropout is not None:
        # Apply dropout to p_attn
        p_attn = dropout(p_attn)
        mainlogger.debug("p_attn = %s", p_attn)
    # Return tuple containing the matrix product of p_attn and parameter value, p_attn
    attn_prod = torch.matmul(p_attn, value), p_attn
    mainlogger.debug("attn_prod = %s", attn_prod)
    return attn_prod

class MultiHeadedAttention(nn.Module):
    """
    Implements 'Multi-Head Attention' proposed in the paper.
    """
    # Default constructor for MultiHeadedAttention
    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        """
        # Create object of superclass torch.nn.Module
        super(MultiHeadedAttention, self).__init__()
        # We assume d_v always equals d_k
        assert d_model % h == 0
        # Set MultiheadedAttention.d_k to be the result of floor division between parameters d_model and h
        self.d_k = d_model // h
        # Set MultiheadedAttention.h to be parameter h
        self.h = h
        # Make an empty torch.Tensor with size given by parameter d_model x d_model
        # A linear transformation is applied to empty tensor with bias sqrt(d_model)
        # Make a torch.nn.Modulelist consisting of 4 clones of the transformed tensor
        # Set MultiheadedAttention.linears to be this ModuleList
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # Set MultiheadedAttention.attn to be None
        self.attn = None
        # Set MultiheadedAttention.dropout based on parameter dropout (default 0.1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Pass Tensor through multiheaded attention technique.
        """
        mainlogger.info("Calling MultiHeadedAttention.forward()")
        # If a mask is specified
        if mask is not None:
            # Same mask applied to all h heads.
            # Change dimensionality of mask
            mask = mask.unsqueeze(1)
        # Set nbatches to be the size of the first ??? of parameter query
        nbatches = query.size(0)
        mainlogger.debug("nbatches = %s", nbatches)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        mainlogger.debug("query = %s", query)
        mainlogger.debug("key = %s", key)
        mainlogger.debug("value = %s", value)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        mainlogger.debug("x = %s", x)
        mainlogger.debug("self.attn = %s", self.attn)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        mainlogger.debug("concatenated x = %s", x)
        x_lin = self.linears[-1](x)
        mainlogger.debug("x_lin = %s", x_lin)
        return x_lin

class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation.
    """
    # Default constructor for PositionwiseFeedForward
    def __init__(self, d_model, d_ff, dropout=0.1):
        mainlogger.debug("Creating PositionwiseFeedForward object with d_model = %s", d_model, ", d_ff = %s", d_ff, ", dropout = %s", dropout)
        # Create object of superclass torch.nn.Module
        super(PositionwiseFeedForward, self).__init__()
        # Set PositionwiseFeedForward.w_1 to be the linear transformed tensor with dimensions given by parameters d_model and d_diff
        # TODO: clarify how Linear() works
        self.w_1 = nn.Linear(d_model, d_ff)
        mainlogger.debug("self.w_1 = %s", self.w_1)
        # Set PositionwiseFeedForward.w_2 to be the linear transformed tensor with dimensions given by paramteters d_ff and d_model
        # TODO: clarify how Linear() works
        self.w_2 = nn.Linear(d_ff, d_model)
        mainlogger.debug("self.w_2 = %s", self.w_2)
        # Set PositionwiseFeedForward.dropout to be based on parameter dropout (default 0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Put Tensor through PositionwiseFeedForward technique. Passes through w_1, then rectified linear unit, then dropout, then w_2.
        """
        mainlogger.info("Calling PositionwiseFeedForward.forward()")
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    """
    Custom version of Embedding lookup table.
    """
    # Default constructor for Embeddings
    def __init__(self, d_model, vocab):
        mainlogger.debug("Creating Embeddings object of vocab = %s", vocab, ", d_model = %s", d_model)
        # Create object of superclass torch.nn.Module
        super(Embeddings, self).__init__()
        # lut => lookup table
        # Set Embeddings.lut to be torch.nn.Embedding with number given by parameter vocab and dimensions given by parameter d_model
        self.lut = nn.Embedding(vocab, d_model)
        mainlogger.debug("Lookup table = %s", self.lut)
        # vocab = 62 d_model = 512
        # Set Embeddings.d_model to be parameter d_model
        self.d_model = d_model

    def forward(self, x):
        """
        Pass tensor through Embeddings. Multiply the lookup table elements by sqrt(dimensions)
        """
        mainlogger.info("Calling Embeddings.forward()")
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    """
    # Default constructor for Positional Encoding
    def __init__(self, d_model, dropout, max_len=5000):
        mainlogger.debug("Creating PositionalEncoding object of d_model = %s", d_model, ", dropout = %s", dropout, ", max_len = %s", max_len)
        # Create object of superclass torch.nn.Module
        super(PositionalEncoding, self).__init__()
        # Set PositionalEncoding.dropout based on parameter dropout
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        
        # Set Tensor pe to be a tensor of zeroes with sizes max_len x d_model
        pe = torch.zeros(max_len, d_model)
        mainlogger.debug("pe = %s", pe)
        # Set Tensor position to be 2D Tensor containing single integer values [0, max_len)
        # Second dimension is only length 1
        # Essentially, this Tensor has been set to be a column vector instead of a row vector
        position = torch.arange(0, max_len).unsqueeze(1)
        mainlogger.debug("position = %s", position)
        # In math notation, div_term = exp(A * -log(10000.0)/d_model)
            # Where A is 1D Tensor containing multiples of 2 in range [0, d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        mainlogger.debug("div_term = %s", div_term)
        # In second dimension, set even-index elements of Tensor pe to be analogous elements of sin(position * div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        mainlogger.debug("pe = %s", pe)
        # In second dimension, set odd-index elements of Tensor pe to be analogous elements of cos(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        mainlogger.debug("pe = %s", pe)
        # Compress pe to 1D Tensor
        pe = pe.unsqueeze(0)
        mainlogger.debug("pe = %s", pe)
        # Save pe as a buffer (not a parameter of the model, but important to track)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Apply the PE function to the input Tensor and apply dropout.
        """
        mainlogger.info("Calling PositionalEncoding.forward()")
        x = x + self.pe[:, :x.size(1)]
        mainlogger.debug("x = %s", x)
        return self.dropout(x)

def make_model(src_vocab, tgt_vocab, N=12, d_model=1024, d_ff=2048, h=8, dropout=0.1):
    """
    Helper: Construct a model from hyperparameters.
    """
    mainlogger.info("Calling make_model()")
    # Set c to be general deep copy operation
    c = copy.deepcopy
    # Set attn to be MultiHeadedAttention Module with given h and d_model from parameters
    attn = MultiHeadedAttention(h, d_model)
    # Set ff to be PositionwiseFeedForwarding Module with given d_model, d_ff, and dropout from parameters
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # Set position to be PositionalEncoding Module with given d_model and dropout from parameters
    position = PositionalEncoding(d_model, dropout)
    # Make model as an EncoderDecoder Module
    model = EncoderDecoder(
        # Pass as argument an Encoder Module operating on EncoderLayer Module with given arguments and N layers
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        # Pass as argument a Decoder Module operating on DecoderLayer Module with given arguments and N layers
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        # Pass as argument a Sequential container (like an ordered ModuleList) with given arguments
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        # Pass as argument a Sequential container with given arguments
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        # Pass as argument a Generator with given arguments
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
    # Default constructor for SimpleLossCompute
    def __init__(self, generator, criterion, opt=None):
        mainlogger.info("Calling SimpleLossCompute as object")        
        # Set data attribute generator to be parameter generator
        self.generator = generator
        # Set data attribute criterion to be parameter criterion
        self.criterion = criterion
        # Set data attribute opt to be parameter opt
        self.opt = opt

    # Define behavior upon function call
    def __call__(self, x, y, norm):
        mainlogger.info("Calling SimpleLossCompute as function")
        # # x = [2,3135,512]
        # # y = [2,3135]
        # Apply generator to input x
        x = self.generator(x)
        # # x = [2,3135,16]
        # Calculate loss
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        # Compute gradient of every parameter of x with grad=true
        loss.backward()
        if self.opt is not None:
            # Perform single optimization step (update parameters)
            self.opt.step()
            # Set the gradients of all optimized torch.Tensor to zero
            # NOTE: opt may already be an optimizer
            self.opt.optimizer.zero_grad()
        # Return the loss * norm as a scalar (loss Tensor should have 1 element)
        return loss.item() * norm