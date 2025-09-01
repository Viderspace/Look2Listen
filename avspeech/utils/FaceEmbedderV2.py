import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List, Optional, Union
from facenet_pytorch import InceptionResnetV1


def chunkate(data: List[np.ndarray],
             chunk_size: int = 75,
             drop_tail: bool = False,
             pad_tail: bool = False,
             max_3_chunks: bool = False) -> Optional[List[List[np.ndarray]]]:
    """
    Split data into fixed-size chunks with different tail handling options.
    """
    if drop_tail and pad_tail:
        raise ValueError("Cannot set both drop_tail and pad_tail to True")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if len(data) == 0:
        return []

    # Handle max_3_chunks constraint first
    if max_3_chunks:
        if len(data) < chunk_size:
            return None
        max_items = 3 * chunk_size
        data = data[:max_items]  # Keep only first 3*chunk_size items

    # Determine chunking parameters
    num_items = len(data)
    num_complete_chunks = num_items // chunk_size
    remainder = num_items % chunk_size

    # Handle tail behavior
    if remainder > 0:
        if drop_tail:
            # Remove remainder items
            data = data[:num_complete_chunks * chunk_size]
        elif pad_tail:
            # Pad with zero items
            if len(data) > 0:
                # Create zero item with same shape and dtype as input items
                sample_shape = data[0].shape
                sample_dtype = data[0].dtype
                zero_item = np.zeros(sample_shape, dtype=sample_dtype)

                # Add zero items to complete the last chunk
                items_to_add = chunk_size - remainder
                zero_items = [zero_item.copy() for _ in range(items_to_add)]
                data.extend(zero_items)

                num_complete_chunks += 1

    # Split into chunks
    chunks = []
    for i in range(num_complete_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = data[start_idx:end_idx]
        chunks.append(chunk)

    if len(chunks) == 0:
        return []

    return chunks


class FaceEmbedderV2:
    """
    PyTorch wrapper for FaceNet face embedding extraction.
    Extracts 1792-dim embeddings from avgpool layer, uses float32 throughout.
    """

    def __init__(self,
                 pretrained: str = 'vggface2',
                 device: Optional[str] = None):
        """
        Initialize the face embedder.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the FaceNet model
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)

        # Storage for extracted features
        self.features = {}

        # Set up hook to extract 1792-dimensional features from avgpool layer
        self._setup_feature_hook()

    def _setup_feature_hook(self):
        """Set up hook to capture avgpool layer output."""

        def hook_fn(name):
            def hook(model, input, output):
                self.features[name] = output

            return hook

        # Register hook on the avgpool layer to get 1792-dim features
        self.model.avgpool_1a.register_forward_hook(hook_fn('avgpool'))

    def _preprocess_face_crop(self, face_crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess single face crop for FaceNet. Handles different input types.
        """
        # Handle different input dtypes to match matplotlib.imread behavior
        if face_crop.dtype in [np.float32, np.float64]:
            # Assume [0,1] range (PNG-like from matplotlib)
            if face_crop.max() <= 1.0:
                image_array = (face_crop * 255).astype(np.uint8)
            else:
                image_array = face_crop.astype(np.uint8)
        else:
            # Assume uint8 [0,255] range (JPG-like from matplotlib)
            image_array = face_crop.astype(np.uint8)

        # Convert to PIL Image
        image = Image.fromarray(image_array)

        # Resize to 160x160 as expected by FaceNet
        image = image.resize((160, 160))

        # Convert to tensor and normalize to [0, 1] (matches original script behavior)
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32)
        image_tensor = image_tensor / 255.0  # [0, 1] range
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 160, 160)

        return image_tensor.to(self.device)

    def _extract_embedding(self, face_crop: np.ndarray) -> torch.Tensor:
        """
        Extract 1792-dim embedding from single face crop using avgpool hook.
        """
        # Preprocess the face crop
        image_tensor = self._preprocess_face_crop(face_crop)

        # Forward pass to trigger hook
        with torch.no_grad():
            _ = self.model(image_tensor)

        # Extract embedding from avgpool layer
        if 'avgpool' in self.features:
            embedding = self.features['avgpool']
            # Flatten to 1D tensor and ensure float32
            embedding = embedding.squeeze().to(torch.float32)
            self.features.clear()  # Clear for next iteration
            return embedding
        else:
            raise RuntimeError("Could not extract avgpool features")

    def embed_75_frames(self, face_crops: List[np.ndarray]) -> torch.Tensor:
        """
        Extract embeddings from exactly 75 face crops. Returns tensor of shape (75, 1792).
        """
        if len(face_crops) != 75:
            raise ValueError(f"Expected exactly 75 face crops, got {len(face_crops)}")

        embeddings = []

        for i, face_crop in enumerate(face_crops):
            try:
                embedding = self._extract_embedding(face_crop)
                embeddings.append(embedding)

            except Exception as e:
                # Return zero embedding if processing fails
                zero_embedding = torch.zeros(1792, dtype=torch.float32, device=self.device)
                embeddings.append(zero_embedding)

        # Stack into tensor of shape (75, 1792)
        embeddings_tensor = torch.stack(embeddings, dim=0)
        return embeddings_tensor

    def compute_embeddings(self,
                           face_crops: List[np.ndarray],
                           drop_tail: bool = False,
                           pad_tail: bool = False,
                           max_3_chunks: bool = False) -> Optional[List[torch.Tensor]]:
        """
        Process any number of face crops by chunking into 75-frame groups.
        """
        # Use chunkate helper to handle all chunking logic
        chunks = chunkate(face_crops,
                          chunk_size=75,
                          drop_tail=drop_tail,
                          pad_tail=pad_tail,
                          max_3_chunks=max_3_chunks)

        # Handle None return from chunkate
        if chunks is None:
            return None

        if len(chunks) == 0:
            return []

        # Process each chunk through the embedding model
        chunk_embeddings = []
        for i, chunk in enumerate(chunks):
            embeddings = self.embed_75_frames(chunk)  # Shape: (75, 1792)
            chunk_embeddings.append(embeddings)

        return chunk_embeddings

    def embed_single_frame(self, face_crop: np.ndarray) -> torch.Tensor:
        """
        Extract embedding from a single face crop.

        Args:
            face_crop: Face crop as numpy array of shape (H, W, 3)

        Returns:
            1792-dimensional embedding as torch.Tensor of shape (1792,) in float32
        """
        return self._extract_embedding(face_crop)


    def get_embeddings_as_numpy(self, embeddings: List[torch.Tensor]) -> List[np.ndarray]:
        """
        Convert list of torch embedding tensors to numpy arrays.

        Args:
            embeddings: List of torch.Tensor, each of shape (75, 1792)

        Returns:
            List of numpy arrays, each of shape (75, 1, 1792) with float32 dtype
            (matches original script's output format)
        """
        numpy_embeddings = []

        for emb_tensor in embeddings:
            # Convert (75, 1792) -> (75, 1, 1792) to match original script format
            emb_np = emb_tensor.cpu().numpy().astype(np.float32)
            emb_np = emb_np[:, np.newaxis, :]  # Add middle dimension
            numpy_embeddings.append(emb_np)

        return numpy_embeddings

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
                'model_type'             : 'InceptionResnetV1',
                'pretrained'             : 'vggface2',
                'device'                 : str(self.device),
                'embedding_dim'          : 1792,
                'expected_input_size'    : (160, 160),
                'input_format'           : 'RGB',
                'preprocessing'          : '[0, 1] range (matches original script)',
                'output_dtype'           : 'float32',
                'compatible_input_dtypes': 'uint8 [0,255], float32/float64 [0,1]'
        }