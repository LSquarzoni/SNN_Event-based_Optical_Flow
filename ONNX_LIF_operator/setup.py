from setuptools import setup, Extension
import os

conda_prefix = os.environ.get("CONDA_PREFIX", "")
torch_site = os.path.join(conda_prefix, "lib", "python3.9", "site-packages", "torch")
include_dirs = [
    os.path.join(torch_site, "include"),
    os.path.join(torch_site, "include", "torch", "csrc", "api", "include"),
]
library_dirs = [os.path.join(torch_site, "lib")]

lif_op_extension = Extension(
    'lif_op',
    sources=['src/lif_op.cpp'],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=["torch", "torch_cpu", "c10"],
    language='c++',
    extra_compile_args=['-std=c++17'],
    extra_link_args=[f"-Wl,-rpath,{library_dirs[0]}"] if library_dirs else [],
)

setup(
    name='ONNX_LIF_operator',
    version='0.1',
    description='Custom LIF operator for ONNX (stateless and stateful variants)',
    packages=[],
    ext_modules=[lif_op_extension],
    install_requires=['torch', 'onnx', 'numpy'],
    zip_safe=False,
)