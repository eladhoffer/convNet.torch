require 'dpnn'
local model = nn.Sequential()
model:add(nn.SpatialConvolution(3,64,5,5,2,2))

model:add(nn.SpatialMaxPooling(3,3,2,2))
model:add(nn.ReLU())
model:add(nn.SpatialBatchNormalization(64))

model:add(nn.FireModule(64,64,64,64))
model:add(nn.SpatialBatchNormalization(128))


model:add(nn.FireModule(128,64,64,64))
model:add(nn.SpatialMaxPooling(3,3,2,2))
model:add(nn.SpatialBatchNormalization(128))

model:add(nn.FireModule(128,64,128,128))
model:add(nn.SpatialBatchNormalization(256))

model:add(nn.FireModule(256,128,128,128))
model:add(nn.SpatialMaxPooling(3,3,2,2))

model:add(nn.SpatialBatchNormalization(256))
model:add(nn.FireModule(256,128,128,128))
model:add(nn.SpatialBatchNormalization(256))

model:add(nn.FireModule(256,128,128,128))
model:add(nn.SpatialBatchNormalization(256))

model:add(nn.FireModule(256,128,128,128))
model:add(nn.SpatialMaxPooling(3,3,2,2))
model:add(nn.SpatialBatchNormalization(256))


model:add(nn.FireModule(256,128,256,256))
model:add(nn.SpatialBatchNormalization(512))


model:add(nn.FireModule(512,256,512,512))
model:add(nn.SpatialBatchNormalization(1024))
model:add(nn.Dropout(0.5))

model:add(nn.SpatialConvolution(1024,1000,1,1))
model:add(nn.SpatialAveragePooling(5,5,1,1))
model:add(nn.View(-1):setNumInputDims(3))
model:add(nn.LogSoftMax())

return model
