import unittest

from tfrecord_dataset.torch import TFRecordDataset


class TFRecordDatasetTest(unittest.TestCase):

  def test_simple(self):
    dataset = TFRecordDataset('tfrecord_dataset/testdata/test.tfrecord', num_epochs=2)

    num = 0
    for x in dataset:
      num += 1

    self.assertEqual(num, 8)


if __name__ == '__main__':
  unittest.main(verbosity=2)
