# TFRecord reader, writer, and PyTorch Dataset

This library allows reading and writing [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord#tfrecords_format_details) files efficiently in Python, and provides an `IterableDataset` interface for TFRecord files in PyTorch. Both uncompressed and compressed gzip TFRecord are supported.

This library is modified from [`tfrecord`](https://pypi.org/project/tfrecord/), to remove its binding to `tf.Example` and support generic TFRecord data.

## Installation

```shell
pip install tfrecord-dataset
```

<!--
```shell
# Install locally:
git clone https://github.com/chaiko/tfrecord_dataset
cd tfrecord_dataset
pip install --editable .

# Release
python -m build
python -m twine upload dist/*
```
-->

## Usage

### Basic read & write

```python
import tfrecord_dataset as tfr

writer = tfr.TFRecordWriter('test.tfrecord')
writer.write(b'Hello world!')
writer.write(b'This is a test.')
writer.close()

for x in tfr.tfrecord_iterator('test.tfrecord'):
    print(bytes(x))
```

### TFRecordDataset

Use `TFRecordDataset` to read TFRecord files in PyTorch.

```python
import torch
from tfrecord_dataset.torch import TFRecordDataset

dataset = TFRecordDataset('test.tfrecord', transform=lambda x: len(x))
loader = torch.utils.data.DataLoader(dataset, batch_size=2)

data = next(iter(loader))
print(data)
```

#### Data transformation

The reader reads TFRecord payload as bytes. You can pass a callable as the
`transform` argument for parsing the bytes into the desired format, as
shown in the simple example above. You can use such transformation for parsing
serialized structured data, e.g. protobuf, numpy arrays, images, etc.

Here is another example for reading and decoding images:

```python
import cv2

dataset = TFRecordDataset(
    'data.tfrecord',
    transform=lambda x: {'image':  cv2.imdecode(x, cv2.IMREAD_COLOR)})
```

#### Shuffling the data

`TFRecordDataset` can automatically shuffle the data when you provide a queue size.

```python
dataset = TFRecordDataset(..., shuffle_queue_size=1024)
```

#### Index

It's recommended to create an index file for each TFRecord file. Index file must be provided when using multiple workers, otherwise the loader may return duplicate records.
```
python -m tfrecord_dataset.tools.tfrecord2idx <tfrecord path> <index path>
```

### MultiTFRecordDataset

Use `MultiTFRecordDataset` to read multiple TFRecord files. This class samples from given TFRecord files with given probability.

```python
import torch
from tfrecord_dataset.torch import MultiTFRecordDataset

dataset = MultiTFRecordDataset(
    data_pattern='test-{}-of-00008',
    index_pattern='test.idx-{}-of-00008',
    splits={'00000': 0.8, '00003': 0.2},
    transform=lambda x: len(x))
loader = torch.utils.data.DataLoader(dataset, batch_size=8)

data = next(iter(loader))
print(data)
```

#### Infinite and finite dataset

By default, `MultiTFRecordDataset` is infinite, meaning that it samples the data forever. You can make it finite by providing the appropriate flag

```python
dataset = MultiTFRecordDataset(..., infinite=False)
```

## Acknowledgements

This repo is forked from https://github.com/vahidk/tfrecord.
