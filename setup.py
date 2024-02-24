import setuptools

with open('README.md') as f:
  long_description = f.read()

setuptools.setup(
    name='tfrecord_dataset',
    version='0.2.0',
    description='TFRecord reader, writer, and PyTorch Dataset.',
    url='https://github.com/chaiko/tfrecord_dataset',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ye Jia',
    author_email='jiayephy@gmail.com',
    license='MIT',
    python_requires='>=3.7',
    install_requires=['numpy', 'crc32c'],
    packages=setuptools.find_packages())
