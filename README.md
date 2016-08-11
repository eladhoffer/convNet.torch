# Convolutional networks using Torch

This is a complete training example for Deep Convolutional Networks on various datasets (ImageNet, Cifar10, Cifar100, STL10, SVHN, MNIST).

It uses TorchNet (<https://github.com/torchnet/torchnet>) for fast data loading and measurements.

It aims to replace both <https://github.com/eladhoffer/ConvNet-torch> and <https://github.com/eladhoffer/ImageNet-Training>

Multiple GPUs are also supported by using nn.DataParallelTable (<https://github.com/torch/cunn/blob/master/docs/cunnmodules.md>).

## Dependencies

- Torch (<http://torch.ch>)
- torchnet (<https://github.com/torchnet/torchnet>)
- cudnn.torch (<https://github.com/soumith/cudnn.torch>) for faster training (optional)

## Data
- To get Cifar data, use @soumith's repo: https://github.com/soumith/cifar.torch.git
- To get the ILSVRC data, you should register on their site for access: <http://www.image-net.org/>
- All data related functions used for training are available at **data.lua**.

## Model configuration

Network model is defined by writing a

<modelname>.lua file in <code>models</code> folder, and selecting it using the <code>model</code> flag.
The model file must return a trainable network. It can also specify additional training options such optimization regime, input size modifications.</modelname>

e.g for a model file:

```lua
local model = nn.Sequential():add(...)
  model.inputSize = 224
  model.reshapeSize = 256
  model.regime = {
    epoch        = {1,    19,   30,   44,   53  },
    learningRate = {1e-2, 5e-3, 1e-3, 5e-4, 1e-4},
    weightDecay  = {5e-4, 5e-4, 0,    0,    0   }
  }
return model
```


## Training

You can start training using **main.lua** by typing:

```lua
th main.lua -model AlexNet -LR 0.01
```

or if you have 2 gpus availiable,

```lua
th main.lua -model AlexNet -LR 0.01 -nGPU 2 -batchSize 256
```

A more elaborate example continuing a pretrained network and saving intermediate results

```lua
th main.lua -model GoogLeNet_BN -batchSize 64 -nGPU 2 -save GoogLeNet_BN -bufferSize 9600 -LR 0.01 -checkpoint 320000 -weightDecay 1e-4 -load ./pretrainedNet.t7
```

## Output

Training output will be saved to folder defined with `save` flag.


## Additional flags

Flag           |  Default Value  | Description
:------------- | :-------------: | :-------------------------------------------------------------------------------
modelsFolder   |    ./models/    | Models Folder
network        |     AlexNet     | Model file - must return valid network.
LR             |      0.01       | learning rate
LRDecay        |        0        | learning rate decay (in # samples)
weightDecay    |      5e-4       | L2 penalty on the weights
momentum       |       0.9       | momentum
batchSize      |      128,       | batch size
optimization   |      'sgd'      | optimization method
seed           |       123       | torch manual random number generator seed
epoch          |       -1        | number of epochs to train, -1 for unbounded
threads        |        8        | number of threads
type           |     'cuda'      | float or cuda
devid          |        1        | device ID (if using CUDA)
nGPU           |        1        | num of gpu devices used
load           |       ''        | load existing net weights
save           | time-identifier | save directory
evalN'         |     100000      | evaluate every N samples
topK'          |        5        | measure top k error
