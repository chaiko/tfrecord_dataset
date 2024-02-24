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

#### Sharded TFRecords

The following `TFRecordDataset` reads TFRecord data from 8 files in parallel. The name of these 8 files match pattern `data-0000?-of-00008`.

```python
dataset = TFRecordDataset(data@8', transform=lambda x: len(x))
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

`TFRecordDataset` automatically shuffles the data with two mechanisms:

1. It reads data into a buffer, and randomly yield data from this buffer. Setting to buffer to a larger size (`buffer_size`) produces better randomness.

2. For sharded TFRecords, it reads multiple files in parallel. Setting `file_parallelism` to a larger number also produces better randomness.


#### Index

Index files are deprecated since v0.2.0. It's no longer required.

Such index files can be generated with:
```
python -m tfrecord_dataset.tools.tfrecord2idx <tfrecord path> <index path>
```

#### Infinite and finite dataset

By default, `TFRecordDataset` is infinite, meaning that it samples the data forever. You can make it finite by setting `num_epochs`.

```python
dataset = TFRecordDataset(..., num_epochs=2)
```

## Acknowledgements

This repo is forked from https://github.com/vahidk/tfrecord.
