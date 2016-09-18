
require 'cudnn'
require 'cunn'
local SpatialConvolution = cudnn.SpatialConvolution--lib[1]
local SpatialMaxPooling = cudnn.SpatialMaxPooling--lib[2]

-- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
-- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
local features = nn.Sequential()
features:add(SpatialConvolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
features:add(SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
features:add(cudnn.ReLU(true))
features:add(cudnn.SpatialBatchNormalization(64))
features:add(SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
features:add(SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
features:add(cudnn.ReLU(true))
features:add(cudnn.SpatialBatchNormalization(192))
features:add(SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
features:add(cudnn.ReLU(true))
features:add(cudnn.SpatialBatchNormalization(384))
features:add(SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
features:add(cudnn.ReLU(true))
features:add(cudnn.SpatialBatchNormalization(256))
features:add(SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
features:add(SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6
features:add(cudnn.ReLU(true))
features:add(cudnn.SpatialBatchNormalization(256))

local classifier = nn.Sequential()
--classifier:add(nn.Dropout(0.5))
classifier:add(nn.SpatialConvolution(256, 4096 ,6,6))
classifier:add(cudnn.ReLU(true))
classifier:add(cudnn.SpatialBatchNormalization(4096))
--classifier:add(nn.Dropout(0.5))
classifier:add(nn.SpatialConvolution(4096, 4096,1,1))
classifier:add(cudnn.ReLU(true))
classifier:add(cudnn.SpatialBatchNormalization(4096))
classifier:add(nn.Copy('torch.CudaHalfTensor', 'torch.CudaTensor',true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Copy('torch.CudaTensor', 'torch.CudaHalfTensor',true))
classifier:add(nn.SpatialConvolution(4096, 1000,1,1))
classifier:add(nn.View(1000):setNumInputDims(3))
classifier:add(nn.LogSoftMax())

local model = nn.Sequential()


model:add(features):add(classifier)

--remove first gradient input calc
features:get(1).updateGradInput = function(x) return end

model.inputSize = 224
model.reshapeSize = 256
model.inputMean = 128
model.inputStd = 128
model.regime = {
  epoch        = {1, 10, 15, 20,25},
  learningRate = {1e-2, 5e-3, 1e-3, 5e-4, 1e-4},
  weightDecay  = {5e-4, 5e-4, 0 }
}
return model
