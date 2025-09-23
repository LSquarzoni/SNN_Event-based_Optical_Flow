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
 ┃ 
 ┣ 📜eval_flow.py
 ┣ 📜train_flow.py
 ┗ 📜quant_model_export.py
```
The folders are quite self-explanatory: \
<span style="color:orange">**/config**</span> contains the configuration files for training and evaluation of the models, \
<span style="color:orange">**/dataloader**</span> has the dataloader code, \
<span style="color:orange">**/loss**</span> has a python script for the loss and errors computation, \
<span style="color:orange">**/models**</span> contains all the building blocks and the definition of the architectures used \
and finally, <span style="color:orange">**/tools**</span> and <span style="color:orange">**/utils**</span> have the scripts for visualization, comparison, logging, etc...

The scripts present in the main folder are used to train and evaluate the models, or to print and export them for other uses. 



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


