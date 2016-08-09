require 'nn'


local model = nn.Sequential()
local feats = nn.Sequential()
local classifier = nn.Sequential()

-- Convolution Layers
feats:add(nn.SpatialConvolution(3, 192, 5, 5,1,1,2,2 ))
feats:add(nn.ReLU(true))
feats:add(nn.SpatialBatchNormalization(192))
feats:add(nn.SpatialConvolution(192, 160,1,1 ))
feats:add(nn.ReLU(true))
feats:add(nn.SpatialBatchNormalization(160))
feats:add(nn.SpatialConvolution(160,96, 1,1 ))
feats:add(nn.ReLU(true))
feats:add(nn.SpatialMaxPooling(3, 3,2,2))
feats:add(nn.SpatialBatchNormalization(96))
feats:add(nn.Dropout(0.5))
feats:add(nn.SpatialConvolution(96,192, 5,5,1,1,2,2 ))
feats:add(nn.ReLU(true))
feats:add(nn.SpatialBatchNormalization(192))
feats:add(nn.SpatialConvolution(192,192, 1,1 ))
feats:add(nn.ReLU(true))

feats:add(nn.SpatialBatchNormalization(192))
feats:add(nn.SpatialConvolution(192,192, 1,1 ))
feats:add(nn.ReLU(true))
feats:add(nn.SpatialMaxPooling(3,3,2,2))
feats:add(nn.SpatialBatchNormalization(192))
feats:add(nn.Dropout(0.5))
feats:add(nn.SpatialConvolution(192,192, 3,3 ,1,1,1,1))
feats:add(nn.ReLU(true))
feats:add(nn.SpatialBatchNormalization(192))
feats:add(nn.SpatialConvolution(192,192, 1,1 ))
feats:add(nn.ReLU(true))
feats:add(nn.Dropout(0.5))
feats:add(nn.SpatialBatchNormalization(192))
classifier:add(nn.SpatialConvolution(192,10, 1,1 ))

classifier:add(nn.SpatialAveragePooling(7,7))
classifier:add(nn.View(10))
classifier:add(nn.LogSoftMax())

model:add(feats):add(classifier)
return model
