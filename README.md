# TFRecord reader, writer, and PyTorch support

This library allows reading and writing [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord#tfrecords_format_details) files efficiently in python. The library also provides an IterableDataset reader of TFRecord files for PyTorch. Currently uncompressed and compressed gzip TFRecords are supported.

## Installation

<!--
```pip3 install tfrecord```
-->

```shell
git clone https://github.com/chaiko/tfrecord
cd tfrecord
pip install --editable ./
```

## Usage

### Basic read & write

```python
import tfrecord

writer = tfrecord.TFRecordWriter('test.tfrecord')
writer.write(b'Hello world!')
writer.write(b'This is a test.')
writer.close()

for x in tfrecord.tfrecord_iterator('test.tfrecord'):
    print(bytes(x))
```

### `TFRecordDataset`

Use `TFRecordDataset` to read TFRecord files in PyTorch.

```python
import torch
from tfrecord.torch import TFRecordDataset

dataset = TFRecordDataset('test.tfrecord', transform=lambda x: len(x))
loader = torch.utils.data.DataLoader(dataset, batch_size=2)

data = next(iter(loader))
print(data)
```

#### Transforming input data

The reader reads TFRecord payloads as bytes. You can optionally pass a callable
to the `transform` argument for post processing into the desired format, as
shown in the example above.

Here is another example for reading and decoding images:

```python
import tfrecord
import cv2

def decode_image(data):
    # get BGR image from bytes
    return {'image':  cv2.imdecode(data, cv2.IMREAD_COLOR)}

dataset = tfrecord.torch.TFRecordDataset(
    'data.tfrecord', transform=decode_image)

data = next(iter(dataset))
print(data)
```

#### Shuffling the data

`TFRecordDataset` can automatically shuffle the data when you provide a queue size.
```
dataset = TFRecordDataset(..., shuffle_queue_size=1024)
```

#### Index

It's recommended to create an index file for each TFRecord file. Index file must be provided when using multiple workers, otherwise the loader may return duplicate records.
```
python -m tfrecord.tools.tfrecord2idx <tfrecord path> <index path>
```

### `MultiTFRecordDataset`

Use `MultiTFRecordDataset` to read multiple TFRecord files. This class samples from given TFRecord files with given probability.

```python
import torch
from tfrecord.torch import MultiTFRecordDataset

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
```
dataset = MultiTFRecordDataset(..., infinite=False)
```

## Acknowledge

This repo is forked from https://github.com/vahidk/tfrecord.
