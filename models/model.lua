require 'nn'
local model = nn.Sequential()

-- Convolution Layers
model:add(nn.SpatialConvolution(3, 64, 5, 5 ))
model:add(nn.SpatialMaxPooling(2, 2))
model:add(nn.ReLU(true))
model:add(nn.SpatialBatchNormalization(64))

model:add(nn.SpatialConvolution(64, 128, 3, 3))
model:add(nn.SpatialMaxPooling(2, 2))
model:add(nn.ReLU(true))

model:add(nn.SpatialBatchNormalization(128))
model:add(nn.SpatialConvolution(128, 256, 3, 3))
model:add(nn.SpatialMaxPooling(2, 2))
model:add(nn.ReLU(true))


--model:add(nn.Dropout(0.5))
model:add(nn.SpatialBatchNormalization(256))
model:add(nn.SpatialConvolution(256, 128, 2,2))
model:add(nn.ReLU(true))
model:add(nn.View(128))

model:add(nn.BatchNormalization(128))
model:add(nn.Dropout(0.5))

model:add(nn.Linear(128,10))
model:add(nn.LogSoftMax())
return model
