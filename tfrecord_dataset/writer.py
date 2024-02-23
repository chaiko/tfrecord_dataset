"""Writer utils."""

import io
import struct

import crc32c
import numpy as np


class TFRecordWriter:
  """Opens a TFRecord file for writing.

    Params:
    -------
    data_path: str
        Path to the tfrecord file.
    """

  def __init__(self, data_path: str) -> None:
    self.file = io.open(data_path, "wb")

  def close(self) -> None:
    """Close the tfrecord file."""
    self.file.close()

  def write(self, record: bytes) -> None:
    """Writes bytes into tfrecord file."""
    length = len(record)
    length_bytes = struct.pack("<Q", length)
    self.file.write(length_bytes)
    self.file.write(TFRecordWriter.masked_crc(length_bytes))
    self.file.write(record)
    self.file.write(TFRecordWriter.masked_crc(record))

  @staticmethod
  def masked_crc(data: bytes) -> bytes:
    """CRC checksum."""
    mask = 0xa282ead8
    crc = crc32c.crc32(data)
    masked = ((crc >> 15) | (crc << 17)) + mask
    masked = np.uint32(masked & np.iinfo(np.uint32).max)
    masked_bytes = struct.pack("<I", masked)
    return masked_bytes
