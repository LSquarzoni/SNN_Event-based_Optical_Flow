# Spiking Neural Networks for Ultra Low Power Event-Based Optical Flow Estimation

Repository structure:

```
📦Event_Flow
 ┣ 📂configs
 ┣ 📂dataloader
 ┣ 📂loss
 ┣ 📂models
 ┣ 📂tools
 ┣ 📂utils
 ┣ 📂ONNX_LIF_operator
 ┃ 
 ┣ 📜eval_flow.py
 ┣ 📜train_flow.py
 ┗ 📜quant_model_export.py
```
The folders are quite self-explanatory: \
:file_folder: /config contains the configuration files for training and evaluation of the models, \
:file_folder: /dataloader has the dataloader code, \
:file_folder: /loss has a python script for the loss and errors computation, \
:file_folder: /models contains all the building blocks and the definition of the architectures used, \
:file_folder: /tools and :file_folder: /utils have the scripts for visualization, comparison, logging, etc... \
Finally, :file_folder: /ONNX LIF operator holds the C++ LIF kernel that makes our LIF operator compatible with ONNX exporting.

The scripts present in the main folder are used to train and evaluate the models, or to print and export them for other uses. 


## Environment setup
I personally worked using a conda environment to manage python packages. \
The packages required to run the code are listed in the requirement.txt file, most of them require pip to be installed in the env. 

```
conda create -n <env-name> python=3.11.13
conda activate <env-name>
conda install pip
pip install -r requirements.txt
```

The last command has to be called from the main folder of the repository, where requirements.txt is located, and withing the right conda env. 



## Python training and evaluation of the models



## ONNX LIF layer generation and usage
The generation of a new operator in ONNX requires following specific steps and to set the environment correctly. \
In my specific case I worked on Ubuntu 24.04.3 LTS. It is mandatory to match the libtorch and pytorch versions in the envirorments; to be sure not to get any compatibility issue, I personally worked in a separate conda environment, based on python 3.11. \
For example, if you want to use the same versions as me:
```
// pytorch
conda create -n <env-name2> python=3.11
conda activate <env-name2>
pip install torch==2.9.1+cpu

// libtorch
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.9.1%2Bcpu.zip
unzip libtorch-shared-with-deps-2.9.1+cpu.zip
rm libtorch-shared-with-deps-2.9.1+cpu.zip
```
onnxruntime needs to be installed as well: \
from the github page (https://github.com/microsoft/onnxruntime/releases), look for the correct version compatible with your system. 

The folder contains already the correctly built operator, but in case it was necessary to make any kind of change to it, I'll explain the way of doing it: \
it is required to work inside the previously introduced conda environment, because at the end the operator will need to be installed inside pytorch, to make the exporting of the layer possible; 
```
cd /ONNX_LIF_operator/build/
rm -rf *
cmake ..
make
```
This will allow to re-built the kernel located in /ONNX_LIF_operator/src/lif_op.cpp. 

To install it into your pytorch and onnx libraries, you will have to call the setup script; inside this file it is important to change the path to your specific libraries' directories.
```
conda activate <env-name2>
cd /ONNX_LIF_operator/
python setup.py install
```
At this point, the kernel will work fine, if correctly called within the python script of your model. \
Specifically, you will have to use the generated kernel instead of the snn torch module at the moment of exportation to ONNX.

## Original work

The original code has been developed by:

```
@article{hagenaarsparedesvalles2021ssl,
  title={Self-Supervised Learning of Event-Based Optical Flow with Spiking Neural Networks},
  author={Hagenaars, Jesse and Paredes-Vall\'es, Federico and de Croon, Guido},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

For a comparison reason, the whole project has been carried out keeping as most things as possible similar to what the authors did, like training specifications, datasets and metrics.
