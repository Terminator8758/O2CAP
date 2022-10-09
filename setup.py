from setuptools import setup, find_packages


setup(name='O2CAP',
      version='1.0.0',
      description='',
      install_requires=[
          'numpy', 'torch', 'torchvision'],
          #'six', 'h5py', 'Pillow', 'scipy',
          #'scikit-learn', 'metric-learn', 'faiss'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Contrastive Learning',
          'Object Re-identification'
      ])
