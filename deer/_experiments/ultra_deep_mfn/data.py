from typing import Tuple, Optional, Any
from abc import abstractproperty, abstractmethod
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


class Dataset:
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, i: int) -> Any:
        pass

    @abstractmethod
    def getcoords(self) -> jnp.ndarray:
        # get the coordinate with the original shape
        pass
    
    @abstractmethod
    def denorm(self, y: jnp.ndarray) -> jnp.ndarray:
        # denormalize the output
        pass

    @abstractproperty
    def ncoords(self) -> int:
        pass

    @abstractproperty
    def nchannels(self) -> int:
        pass

class ImageDataset(Dataset):
    def __init__(self, image_path: str, rngkey: jax.random.PRNGKey, maxnpts: Optional[int] = None,
                 dtype: jnp.dtype = jnp.float32):
        # load the image and normalize the value to -1 to 1
        self.image = np.array(Image.open(image_path).convert('RGB')) / 127.5 - 1  # shape: (height, width, channel)
        height, width = self.image.shape[:2]

        # create the coordinate of the image pixel that's valued from -1 to 1 using meshgrid
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        y, x = np.meshgrid(y, x, indexing="ij")  # shape: (height, width)
        # stack the coordinate
        self.coord = np.stack([x, y], axis=-1)  # shape: (height, width, 2)
        self.meshshape = (height, width)

        # flatten the image and coord
        self.image = self.image.reshape(-1, self.image.shape[-1])  # shape: (height * width, channel)
        self.coord = self.coord.reshape(-1, self.coord.shape[-1])  # shape: (height * width, 2)

        # convert to jax.numpy array
        self.image = jnp.array(self.image, dtype=dtype)
        self.coord = jnp.array(self.coord, dtype=dtype)

        self.key = rngkey
        self.maxnpts = maxnpts

    def __len__(self) -> int:
        return 1

    def __getitem__(self, i: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.maxnpts is None:
            return self.coord, self.image
        else:
            # randomly sample the points
            self.key, subkey = jax.random.split(self.key)
            idx = jax.random.permutation(subkey, jnp.arange(self.coord.shape[0]))[:self.maxnpts]
            return self.coord[idx], self.image[idx]

    def getcoords(self) -> jnp.ndarray:
        return self.coord.reshape(*self.meshshape, self.coord.shape[-1])

    @property
    def ncoords(self) -> int:
        return self.coord.shape[-1]

    @property
    def nchannels(self) -> int:
        return self.image.shape[-1]

    def denorm(self, y: jnp.ndarray) -> jnp.ndarray:
        # convert the image from -1 to 1 to 0 to 255 and convert to uint8
        return jnp.clip((y + 1) * 127.5, 0, 255).astype(jnp.uint8)
