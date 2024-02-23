"""Reader utils."""

import functools
import gzip
import io
import os
import struct
import typing

import numpy as np

from tfrecord_dataset import iterator_utils


def tfrecord_iterator(
    data_path: str,
    index_path: typing.Optional[str] = None,
    shard: typing.Optional[typing.Tuple[int, int]] = None,
    compression_type: typing.Optional[str] = None,
) -> typing.Iterable[bytes]:
  """Create an iterator over the tfrecord dataset.

    Params:
    -------
    data_path: str
        TFRecord file path.

    index_path: str, optional, default=None
        Index file path. Can be set to None if no file is available.

    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).

    Yields:
    -------
    datum_bytes: bytes
        Object referencing the specified `datum_bytes` contained in the
        file (for a single record).
    """
  if compression_type == "gzip":
    file = gzip.open(data_path, "rb")
  elif compression_type is None:
    file = io.open(data_path, "rb")
  else:
    raise ValueError("compression_type should be either 'gzip' or None")
  length_bytes = bytearray(8)
  crc_bytes = bytearray(4)
  datum_bytes = bytearray(1024 * 1024)

  def read_records(start_offset=None, end_offset=None):
    nonlocal length_bytes, crc_bytes, datum_bytes

    if start_offset is not None:
      file.seek(start_offset)
    if end_offset is None:
      end_offset = os.path.getsize(data_path)
    while file.tell() < end_offset:
      if file.readinto(length_bytes) != 8:
        raise RuntimeError("Failed to read the record size.")
      if file.readinto(crc_bytes) != 4:
        raise RuntimeError("Failed to read the start token.")
      length, = struct.unpack("<Q", length_bytes)
      if length > len(datum_bytes):
        datum_bytes = datum_bytes.zfill(int(length * 1.5))
      datum_bytes_view = memoryview(datum_bytes)[:length]
      if file.readinto(datum_bytes_view) != length:
        raise RuntimeError("Failed to read the record.")
      if file.readinto(crc_bytes) != 4:
        raise RuntimeError("Failed to read the end token.")
      yield bytes(datum_bytes_view)

  if index_path is None:
    yield from read_records()
  else:
    index = np.loadtxt(index_path, dtype=np.int64)[:, 0]
    if shard is None:
      offset = np.random.choice(index)
      yield from read_records(offset)
      yield from read_records(0, offset)
    else:
      num_records = len(index)
      shard_idx, shard_count = shard
      start_index = (num_records * shard_idx) // shard_count
      end_index = (num_records * (shard_idx + 1)) // shard_count
      start_byte = index[start_index]
      end_byte = index[end_index] if end_index < num_records else None
      yield from read_records(start_byte, end_byte)

  file.close()


def multi_tfrecord_iterator(
    data_pattern: str,
    index_pattern: typing.Union[str, None],
    splits: typing.Dict[str, float],
    compression_type: typing.Optional[str] = None,
    infinite: bool = True,
) -> typing.Iterable[bytes]:
  """Create an iterator by reading and merging multiple tfrecord datasets.

    NOTE: Sharding is currently unavailable for the multi tfrecord loader.

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

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    infinite: bool, optional, default=True
        Whether the returned iterator should be infinite or not

    Returns:
    --------
    it: iterator
        A repeating iterator that generates batches of data.
    """
  iters = [
      functools.partial(
          tfrecord_iterator,
          data_path=data_pattern.format(split),
          index_path=index_pattern.format(split) if index_pattern is not None else None,
          compression_type=compression_type) for split in splits.keys()
  ]
  return iterator_utils.sample_iterators(iters, list(splits.values()), infinite=infinite)
