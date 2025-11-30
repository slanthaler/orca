from pathlib import Path
import numpy as np
import torch

def _torch_tensors(*arrays):
    '''
    Converts numpy arrays to torch tensors.
    '''
    return tuple(torch.tensor(arr, dtype=torch.float32) for arr in arrays)

def np_data_to_torch_data(data_np, batch_size):
    '''
    Converts numpy dataset to PyTorch DataLoaders.
    '''
    _TensorDataset = lambda arrays: torch.utils.data.TensorDataset(*_torch_tensors(*arrays))

    #
    train_ds = _TensorDataset(data_np['train'])
    val_ds = _TensorDataset(data_np['val'])
    test_ds = _TensorDataset(data_np['test'])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return {
        'input_dim': data_np['input_dim'], 'output_dim': data_np['output_dim'],
        'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader
    }


def default_dataloader():
    '''
    Loads a toy linear dataset.
    '''
    data_dir = Path(__file__).parent.parent / 'data/linear/'
    data_train = np.load(data_dir / 'linear_train.npz', allow_pickle=True)
    data_test = np.load(data_dir / 'linear_test.npz', allow_pickle=True)

    # 
    input_dim = data_train['x_train'].shape[1]
    output_dim = data_train['y_train'].shape[1]

    # train / val split
    n_train = len(data_train['x_train'])
    
    # shuffle before split
    shuffle_idx = np.random.permutation(n_train)
    data_train = (data_train['x_train'][shuffle_idx], data_train['y_train'][shuffle_idx])

    # split off validation samples
    split_idx = int(0.8 * n_train)
    data_val = (data_train[0][split_idx:], data_train[1][split_idx:])
    data_train = (data_train[0][:split_idx], data_train[1][:split_idx])

    # same tuple format for test
    data_test = (data_test['x_test'], data_test['y_test'])

    # package this
    data = {'input_dim': input_dim, 'output_dim': output_dim,
            'train': data_train, 'val': data_val, 'test': data_test}
    
    return data