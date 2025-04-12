import pickle
from torch.utils.data import DataLoader
from preprocess import SVHNDataset, transform


def load_dataloader_from_artifact(artifact_path, batch_size=None, shuffle=None):
    """Load a dataloader from a saved artifact."""
    with open(artifact_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Override batch_size and shuffle if provided
    batch_size = batch_size if batch_size is not None else metadata['batch_size']
    shuffle = shuffle if shuffle is not None else metadata['shuffle']
    
    dataset = SVHNDataset(metadata['images'], metadata['labels'], transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

