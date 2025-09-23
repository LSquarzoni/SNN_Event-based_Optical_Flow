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
/config contains the configuration files for training and evaluation of the models, \
/dataloader has the dataloader code, \
/loss has a python script for the loss and errors computation, \
/models contains all the building blocks and the definition of the architectures used \
and finally, /tools and /utils have the scripts for visualization, comparison, logging, etc...

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



## Python training and evaluation of the models



## ONNX LIF layer generation and usage


