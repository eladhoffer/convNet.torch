require 'nn'


local model = nn.Sequential()

-- Convolution Layers
model:add(nn.SpatialConvolution(3, 192, 5, 5,1,1,2,2 ))
model:add(nn.ReLU(true))
model:add(nn.SpatialBatchNormalization(192))
model:add(nn.SpatialConvolution(192, 160,1,1 ))
model:add(nn.ReLU(true))
model:add(nn.SpatialBatchNormalization(160))
model:add(nn.SpatialConvolution(160,96, 1,1 ))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(3, 3,2,2))
model:add(nn.SpatialBatchNormalization(96))
model:add(nn.Dropout(0.5))
model:add(nn.SpatialConvolution(96,192, 5,5,1,1,2,2 ))
model:add(nn.ReLU(true))
model:add(nn.SpatialBatchNormalization(192))
model:add(nn.SpatialConvolution(192,192, 1,1 ))
model:add(nn.ReLU(true))

model:add(nn.SpatialBatchNormalization(192))
model:add(nn.SpatialConvolution(192,192, 1,1 ))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(3,3,2,2))
model:add(nn.SpatialBatchNormalization(192))
model:add(nn.Dropout(0.5))
model:add(nn.SpatialConvolution(192,192, 3,3 ,1,1,1,1))
model:add(nn.ReLU(true))
model:add(nn.SpatialBatchNormalization(192))
model:add(nn.SpatialConvolution(192,192, 1,1 ))
model:add(nn.ReLU(true))
model:add(nn.Dropout(0.5))
model:add(nn.SpatialBatchNormalization(192))
model:add(nn.SpatialConvolution(192,10, 1,1 ))

model:add(nn.SpatialAveragePooling(7,7))
model:add(nn.View(10))
model:add(nn.LogSoftMax())


return model
