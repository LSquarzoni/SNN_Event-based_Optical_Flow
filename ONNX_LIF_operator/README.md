# ONNX LIF Operator

This project implements a custom Leaky Integrate-and-Fire (LIF) operator for ONNX, designed to facilitate the integration of spiking neural networks with ONNX-compatible frameworks. The operator is implemented in C++ and provides a Python interface for ease of use within PyTorch.

## Project Structure

The project consists of the following files and directories:

- **src/**: Contains the source code for the LIF operator.
  - **lif_op.cpp**: C++ implementation of the custom LIF operator, including forward and backward passes.
  - **lif_op.h**: Header file declaring the functions and classes used in `lif_op.cpp`.
  - **python/**: Contains the Python wrapper for the C++ LIF operator.
    - **lif_op.py**: Python interface to the LIF operator.

- **test/**: Contains unit tests for the LIF operator.
  - **test_lif_op.py**: Tests the functionality of the LIF operator.

- **CMakeLists.txt**: Configuration file for CMake to build the C++ components.

- **setup.py**: Setup script for the Python package, defining metadata and dependencies.

- **requirements.txt**: Lists the Python dependencies required for the project.

## Installation

To install the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ONNX_LIF_operator
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Build the C++ extension:
   ```
   python setup.py install
   ```

## Usage

Once installed, you can use the LIF operator in your PyTorch models. Import the operator from the Python wrapper:

```python
from lif_op import LIF
```

You can then create an instance of the LIF operator and use it in your neural network.

## Testing

To run the unit tests for the LIF operator, execute the following command:

```
pytest test/test_lif_op.py
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.