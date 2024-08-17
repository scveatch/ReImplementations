"""
A module that implements "An Image is worth 16x16 words" by Dosovitskiy et al. 

"""

import math # GCD, sqrt
import torch  # For tensor operations and ML
from torch import nn # For layers
import torch.utils.data  # Dataloaders
from torchvision import datasets, transforms  # For MNIST data; transformations
import matplotlib.pyplot as plt # Visualizations


# Helper Functions
##################################
def show(loader):
    """
    Extracts a single image from a torch dataloader object and displays it.
    Args:
        Loader :  A torch Dataloader object
    Returns :
        None
    """
    single_image = next(iter(loader))[0][0]
    plt.imshow(single_image.permute(1, 2, 0))
    plt.show()

def perfect_square(n):
    """
    Checks if an input is a perfect square 
    Args: 
        n : an integer input
    Returns:
        Boolean
    """
    return int(math.sqrt(n))**2 == n

def load_divisors(w, h):
    """
    Extract possible values for square patch size given input image width and height
    Args:
        w : integer picture width
        h : integer picture height
    Returns: 
        divs : A sorted list of possible values
    """
    gcd = math.gcd(w * h)
    divs = set()
    for i in range(2, int(math.sqrt(gcd)) + 1):
        if gcd % i == 0 and perfect_square(i):
            divs.add(i)
            divs.add(gcd // i)
    return sorted(divs)

# Model Functions
##################################

def patch(images, n_patches):
    """
    Takes an input image "x" and splits it into square patches
    as defined by "n_patches"

    Args:
        images (torch.tensor): A torch tensor of images organized in (b, c, h, w)
        n_patches (int): The number of patches to generate from the input images

    Raises:
        ValueError: patch function requires square images
        ValueError: Check that n_patches square patches can be made from input and image size. 

    Returns:
        torch.tensor: A torch tensor of image patches in the shape (b, n_patches, h * w * c // n_patches)
    """
    n, c, h, w = images.shape

    if h != w:
        raise ValueError(f"Patch can only be run over square images, got image with dimensions {h}x{w}")
    if n_patches not in load_divisors(w, h):
        raise ValueError(f"Input image of size {h}x{w} cannot be evenly split into {n_patches} parts. Possible values are {load_divisors(h, w)}")
    
    skip = int(math.sqrt(n_patches))
    stride = h // skip
    
    patches = torch.zeros(n, n_patches, h * w * c // n_patches)

    for index, img in enumerate(images):
        patch_index = 0
        for i in range(skip):
            for j in range(skip):
                patch = img[
                    :,
                    i * stride : (i + 1) * stride, # Height selection
                    j * stride : (j + 1) * stride  # Width selection
                ]
                patches[index, patch_index] = patch.flatten()
                patch_index += 1
    return patches

class MHSA():
    """
    Multi-head Self Attention Class. 
    """
    def __init__(self, token_dimension, heads = 2, **kwargs):
        """
        Implements a simplified version of MHSA. 

        Args:
            token_dimension (int): The dimensionality of the token embeddings -- the size of the feature vector
            heads (int, optional): The number of self attention heads in the model. Defaults to 2.

        Raises:
            ValueError: Token dimension cannot be evenly divided by the number of attention heads provided. 
        """
        super(MHSA, self).__init__()
        self.D = token_dimension
        self.heads = heads
        if self.D % self.heads != 0:
            raise ValueError(f"Cannot divide Token dimension {self.D} by {self.heads} attention heads")
        
        self.dim_heads = self.D // self.heads
        # Generate mappings for Query, Key, Value for each number of heads
        self.q_mappings = nn.Module([
            nn.Linear(self.dim_heads, self.dim_heads) for _ in range(self.heads)
        ])
        self.k_mappings = nn.Module([
            nn.Linear(self.dim_heads, self.dim_heads) for _ in range(self.heads)
        ])
        self.v_mappings = nn.Module([
            nn.Linear(self.dim_heads, self.dim_heads) for _ in range(self.heads)
        ])

        self.softmax = nn.Softmax()

    def forward(self, sequences):
        """
        Forward method for MHSA

        Args:
            sequences (torch.tensor): The input tensor containing the token embeddings
            expecting input with shape (batch, num_tokens, token_dim).  

        Returns:
            torch.tensor: The output tensor after applying MHSA, with the same shape as
            input (batch, num_tokens, token_dim). 
        """
        result = []
        for sequence in sequences:
            seq_result = []
            for head in self.heads:
                q_map = self.q_mappings[head]
                k_map = self.k_mappings[head]
                v_map = self.v_mappings[head]

                seq = sequence[:, head * self.dim_heads : (1 + head) * self.dim_heads]
                # Calculate embeddings for the tokens
                q = q_map(seq)
                k = k_map(seq)
                v = v_map(seq)
                # Calculate attention weights (@ computes dot product)
                attention = self.softmax(q @ k.T / math.sqrt(self.dim_heads))
                # Calculate weighted sum 
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim = 0) for r in result]) # return to shape (b, s, t)

class VitBlock():
    def __init__(self, token_dim, heads, mlp_size):
        super(VitBlock, self).__init__()
        self.token_dim = token_dim
        self.heads = heads
        self.mlp_size = mlp_size

        self.norm1 = nn.LayerNorm(self.token_dim)
        self.mhsa = MHSA(self.token_dim, self.heads)
        self.norm2 = nn.LayerNorm(self.token_dim)
        # expand to mlp_size hidden layers, shrink to token_dim
        self.mlp = nn.Sequential([
            nn.Linear(self.token_dim, self.mlp_size * self.token_dim), 
            nn.GELU(),
            nn.Linear(self.token_dim * self.mlp_size, self.token_dim)
        ])
    
    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out 
    
class ViT():
    def __init__(self, chw, n_patches, n_vit_blocks, hidden_d, token_dim, heads = 2, mlp_size = 4, out_d = 10):
        super(ViT, self).__init__()
        self.chw = chw
        self.n_patches = n_patches
        self.n_vit = n_vit_blocks
        self.hidden_d = hidden_d
        self.token_dim = token_dim
        self.heads = heads
        self.mlp_size = mlp_size
        self.out_d = out_d

        sequence_len = chw[0] * chw[1] * chw[2] // n_patches 

        self.linear_proj = (sequence_len, hidden_d)

        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        self.register_buffer(
            "positional embeddings", 
            get_positional_embeddings(sequence_len, self.hidden_d)

        )

def get_positional_embeddings(sequence_len, dimensions):
    result = torch.ones(sequence_len, dimensions)

    for i in range(sequence_len):
        for j in range(dimensions):
            if j % 2 == 0:
                result[i][j] = math.sin(i / 10000**(j / dimensions))
            else:
                result[i][j] = math.cos(i / 10000**(j / dimensions))
    return result





if __name__ == "__main__":
    transform = transforms.ToTensor()
    train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train)
    images = torch.stack([img for batch in train_loader for img in batch[0]]) # create image tensor
    x = patch(images, 4)
    print(x.shape)
    print(x[0].shape)
