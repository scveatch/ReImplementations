
import torch # Tensor operations / ml
from torch import nn # Layers

DEFAULT_ALPHA = 0.5

class Retention(nn.Module):
    """
    Implements the mathematics for the retention mechanism.
    """ 
    def __init__(self, alpha, embedding_dimension, chunk_size, max_len, mode = "parallel"):
        super(Retention, self).__init__()
        self.alpha = alpha
        self.embed_dim = embedding_dimension
        self.chunk_size = chunk_size
        self.mode = "parallel"
        self.max_len = max_len

        # Useful Buffers:
        self.sqrtD = self.register_buffer("sqrtD", torch.tensor(self.embed_dim**0.5)) # sqrt of Dimension
        self.decay_mask = self.register_buffer("decay_mask", torch.tensor([self.alpha ** (i - j) for j in self.max_len] for i in self.max_len).tril())
        self.causal_mask = self.register_buffer("causal_mask", torch.ones(self.max_len, self.max_len).tril())
        rention_mask_chunk = self.register_buffer("retention_mask_chunk", torch.tensor([self.chunk_size - i - 1] for i in range(self.chunk_size)))

        # QKV calculation
        self.qkv = nn.Linear(embedding_dimension, embedding_dimension * 3) # 3 for Q, K, V, respectively

    def forward_recurrent(self, input, state):
        """
        Propagates the image through the model recurrently.

        Args:
            input (torch.tensor): The image input; expected shape:
            batch_size, sequence_length, and dimension
                - batch_size: the number of samples in the batch
                - sequence_length: number of pixels per sample in batch
                - dimension: the dimensionality of the sequence element
            state (torch.tensor): 3D tensor holding the current state across the
            sequence, initalized as a zero tensor with shape (batch_size, dim, dim).

        Returns:
            torch.tensor: A 3D tensor of shape `(batch_size, sequence_length, dim)` representing
            the output for each sequence element in the batch.
        """
        batch_size, sq_len, dim = input.shape

        state = torch.zeros(batch_size, dim, dim).to(input.device)

        outputs = []
        for i in range(sq_len):
            input_sq = input[:, i]
            q, k, v = self.qkv(input_sq).chunk(3, dim = 1)

            state = self.alpha * state + k.T @ v
            ret = (q.unsqueeze(0) @ state) / self.sqrtD
            
            outputs.append(ret.squeeze(0))

        return torch.stack(outputs, dim = 1)
    
    def forward_parallel(self, input):
        """
        Propagates the image through the model in parallel.

        Args:
            input (torch.tensor): 3D tensor representing the image sequences

        Returns:
            torch.tensor: A 3D tensor of shape (batch_size, sq_len, dim) representing 
            the output for each sequence in the batch. 
        """
        batch_size, sq_len, dim = input.shape
        q, k, v = self.qkv(input).chunk(3, dim = 1)

        M = (self.causal_mask[:sq_len, :sq_len] * self.decay_mask[:sq_len, :sq_len]).repeat(batch_size, 1, 1)
        
        attention = q @ k.transpose(-1, -2) / self.sqrtD
        ret = (attention * M) @ v

        return ret
    
    def forward(self, input, state = None, mode = "parallel", chunk_size = None):
        """
        Handles the forward call based on `mode` value.

        Args:
            input (torch.tensor): 3D tensor representing the image sequences
            state (torch.tensor, optional): 3D tensor representing the state matrix. Defaults to None.
            mode (str, optional): Option to change forward method. Defaults to "parallel".
            chunk_size (int, optional): Number of chunks to partition the input into. Defaults to None.

        Raises:
            ValueError: Unknown mode. Rises when mode is not `parallel` or `recurrent`.

        Returns:
            torch.tensor: Processed 3D tensor
        """
        if mode == "parallel":
            return self.forward_parallel(input)
        if mode == "recurrent":
            return self.forward_recurrent(input, state)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
class MultiHeadRetention(nn.Module):
    def __init__(self, heads, alphas, embed_dim, max_len, chunk_size, mode = "parallel"):
        super(MultiHeadRetention, self).__init__()
        self.n_heads = heads
        self.alphas = alphas
        self.dim = embed_dim
        self.max_len = max_len
        self.chunk_size = chunk_size
        self.mode = mode

        if embed_dim % heads != 0:
            raise ValueError(f"Embedding dimension must be wholly divisible by 
                             number of heads. Dim: {embed_dim}; heads: {heads}")
        
        head_dim = embed_dim // heads

        self.heads = nn.ModuleList([
            Retention(head_dim, chunk_size= None, max_len= max_len, mode = mode) for _ in range(heads)
        ])

        

        
    

        

if __name__ == "__main__":
    pass