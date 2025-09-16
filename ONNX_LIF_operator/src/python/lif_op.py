from setuptools import setup, Extension
import numpy as np

# Define the C++ extension
lif_op_extension = Extension(
    'lif_op',
    sources=['../lif_op.cpp'],  # Path to the C++ source file
    include_dirs=[np.get_include()],  # Include NumPy headers
    language='c++'
)

# Setup function
setup(
    name='onnx_lif_operator',
    version='0.1',
    description='Custom LIF operator for ONNX',
    ext_modules=[lif_op_extension],
    install_requires=[
        'torch',
        'onnx',
        'onnxruntime',
        'numpy'
    ],
    python_requires='>=3.6',
)