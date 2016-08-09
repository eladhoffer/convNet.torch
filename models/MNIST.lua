require 'nn'
require 'dpnn'
local model = nn.Sequential()
local feats = nn.Sequential()

-- Convolution Layers
feats:add(nn.SpatialConvolution(1, 32, 5, 5,1,1,1,1 ))
feats:add(nn.SpatialMaxPooling(2, 2))
feats:add(nn.LeakyReLU(true))
feats:add(nn.SpatialBatchNormalization(32))

feats:add(nn.SpatialConvolution(32, 64, 3, 3,1,1,1,1))
feats:add(nn.LeakyReLU(true))
feats:add(nn.SpatialBatchNormalization(64))

feats:add(nn.SpatialConvolution(64, 64, 3, 3,1,1,1,1))
feats:add(nn.SpatialMaxPooling(2, 2))
feats:add(nn.LeakyReLU(true))
feats:add(nn.SpatialBatchNormalization(64))

feats:add(nn.SpatialConvolution(64, 128, 3, 3,1,1,1,1))
feats:add(nn.LeakyReLU(true))
feats:add(nn.SpatialBatchNormalization(128))

local classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.SpatialConvolution(128, 10, 1,1))
classifier:add(nn.SpatialAveragePooling(6,6,1,1))
classifier:add(nn.View(10))
classifier:add(nn.LogSoftMax())


model:add(feats):add(classifier)

return model
