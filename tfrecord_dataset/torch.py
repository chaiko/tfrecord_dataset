"""Load tfrecord files into torch datasets."""

import typing

import numpy as np
import torch.utils.data

from tfrecord_dataset import reader
from tfrecord_dataset import iterator_utils


class TFRecordDataset(torch.utils.data.IterableDataset):
    """Parses a TFRecord dataset into `IterableDataset` object.

    Params:
    -------
    data_path: str
        The path to the tfrecords file.

    index_path: str or None
        The path to the index file.

    shuffle_queue_size: int, optional, default=None
        Length of buffer. Determines how many records are queued to
        sample from.

    transform : a callable, default = None
        A function that takes in the input `bytes`, transforms it and returns a
        desirable output.

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    """

    def __init__(self,
                 data_path: str,
                 index_path: typing.Optional[str] = None,
                 shuffle_queue_size: typing.Optional[int] = None,
                 transform: typing.Callable[[dict], typing.Any] = None,
                 compression_type: typing.Optional[str] = None,
                 ) -> None:
        super().__init__()
        self.data_path = data_path
        self.index_path = index_path
        self.shuffle_queue_size = shuffle_queue_size
        self.transform = transform or (lambda x: x)
        self.compression_type = compression_type

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        else:
            shard = None
        it = reader.tfrecord_iterator(data_path=self.data_path,
                                      index_path=self.index_path,
                                      shard=shard,
                                      compression_type=self.compression_type)
        if self.shuffle_queue_size:
            it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)
        if self.transform:
            it = map(self.transform, it)
        return it


class MultiTFRecordDataset(torch.utils.data.IterableDataset):
    """Parse multiple (generic) TFRecords datasets into an `IterableDataset`
    object, which contain `np.ndarrays`s.

    Params:
    -------
    data_pattern: str
        Input data path pattern.

    index_pattern: str or None
        Input index path pattern.

    splits: dict
        Dictionary of (key, value) pairs, where the key is used to
        construct the data and index path(s) and the value determines
        the contribution of each split to the batch.

    shuffle_queue_size: int, optional, default=None
        Length of buffer. Determines how many records are queued to
        sample from.

    transform : a callable, default = None
        A function that takes in the input `bytes`, transforms it and returns a
        desirable output.

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    infinite: bool, optional, default=True
        Whether the Dataset should be infinite or not
    """

    def __init__(self,
                 data_pattern: str,
                 index_pattern: typing.Optional[str] = None,
                 *,
                 splits: typing.Dict[str, float],
                 shuffle_queue_size: typing.Optional[int] = None,
                 transform: typing.Callable[[dict], typing.Any] = None,
                 compression_type: typing.Optional[str] = None,
                 infinite: bool = True) -> None:
        super().__init__()
        self.data_pattern = data_pattern
        self.index_pattern = index_pattern
        self.splits = splits
        self.shuffle_queue_size = shuffle_queue_size
        self.transform = transform
        self.compression_type = compression_type
        self.infinite = infinite

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        it = reader.multi_tfrecord_iterator(data_pattern=self.data_pattern,
                                            index_pattern=self.index_pattern,
                                            splits=self.splits,
                                            compression_type=self.compression_type,
                                            infinite=self.infinite)
        if self.shuffle_queue_size:
            it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)
        if self.transform:
            it = map(self.transform, it)
        return it
