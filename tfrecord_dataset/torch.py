"""Load tfrecord files into torch datasets."""

import logging
from typing import Any, Callable, Iterator, Optional

import numpy as np
import torch.utils.data

from tfrecord_dataset import reader


class TFRecordDataset(torch.utils.data.IterableDataset):
  """An IterableDataset reads from regular or sharded TFRecord files.

  This class is not thread-safe.
  """

  def __init__(self,
               file_pattern: str,
               transform: Optional[Callable[[dict], Any]] = None,
               file_parallelism: int = 8,
               buffer_size: int = 256,
               num_epochs: Optional[int] = None) -> None:
    """Constructor.

    Args:
      file_pattern: file path or pattern to TFRecord files.
      transform: Transformation to apply on the raw TFRecord data.
      file_parallelism: Number of files to read in parallel.
      buffer_size: The size of the reading buffer.
      num_epochs: Reads this many of epoch. Set None for infinitely reading.
    """
    super().__init__()

    assert '*' not in file_pattern
    assert '?' not in file_pattern

    if '@' in file_pattern:
      stem, num_shards = file_pattern.split('@', 2)
      num_shards = int(num_shards)
      self.filenames = [f'{stem}-{i:05d}-of-{num_shards:05d}' for i in range(num_shards)]
    else:
      self.filenames = [file_pattern]

    self.transform = transform or (lambda x: x)

    self.file_parallelism = file_parallelism
    self.buffer_size = buffer_size

    self.epoch = 0
    self.num_epochs = num_epochs

    self.filename_queue = []
    self.record_buffer = []  # buffer of processed records (tranform applied)
    self.iters = []  # iterators of raw records

  def _read_one_record(self):
    """Reads one record into the buffer and applies the transform."""
    while self.iters:
      idx = np.random.choice(len(self.iters))
      try:
        data = next(self.iters[idx])
      except StopIteration:
        if self.filename_queue:
          filename = self.filename_queue.pop()
          logging.info(f'Processing file: {filename}')
          self.iters[idx] = reader.tfrecord_iterator(filename)
        else:
          del self.iters[idx]
      else:
        self.record_buffer.append(self.transform(data))
        break

  def _init_epoch(self, epoch: int):
    """Initializes epoch and loads buffer."""
    self.epoch = epoch
    logging.info(f'Start epoch {epoch}')

    self.filename_queue = self.filenames.copy()
    np.random.shuffle(self.filename_queue)

    self.iters = []
    while len(self.iters) < self.file_parallelism and self.filename_queue:
      filename = self.filename_queue.pop()
      logging.info(f'Processing file: {filename}')
      self.iters.append(reader.tfrecord_iterator(filename))

    while len(self.record_buffer) < self.buffer_size and self.iters:
      self._read_one_record()

  def __iter__(self) -> Iterator:
    self._init_epoch(epoch=0)
    return self

  def __next__(self):
    if not self.record_buffer:
      if not self.num_epochs or self.epoch + 1 < self.num_epochs:
        self._init_epoch(epoch=self.epoch + 1)
      else:
        raise StopIteration

    self._read_one_record()

    if not self.record_buffer:
      raise StopIteration
    else:
      idx = np.random.choice(len(self.record_buffer))
      record = self.record_buffer[idx]
      self.record_buffer[idx] = self.record_buffer[-1]
      self.record_buffer.pop()
      return record
