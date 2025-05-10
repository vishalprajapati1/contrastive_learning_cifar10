import torch
import numpy as np

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'mps'
    if torch.backends.mps.is_available() else 'cpu'
)

def from_numpy(x: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype).to(device)

def to_numpy(x: torch.Tensor) -> np.ndarray:
    if x.device.type == 'cuda':
        x = x.cpu()  # Move tensor to CPU before converting to numpy
    with torch.no_grad():
        return x.numpy()