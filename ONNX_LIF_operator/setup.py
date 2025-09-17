from setuptools import setup, Extension
import numpy as np


lif_op_extension = Extension(
    'lif_op',
    sources=['src/lif_op.cpp'],
    include_dirs=["/home/lorenzo/onnxruntime/include",
                  "/home/lorenzo/libtorch/include",
                  "/home/lorenzo/libtorch/include/torch/csrc/api/include"],
    language='c++',
)

setup(
    name='ONNX_LIF_operator',
    version='0.1',
    description='Custom LIF operator for ONNX',
    packages=['src.python'],
    ext_modules=[lif_op_extension],
    install_requires=[
        'torch',
        'onnx',
        'onnxruntime',
        'numpy',
    ],
    zip_safe=False,
)