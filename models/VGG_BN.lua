--borrowed from https://raw.githubusercontent.com/soumith/imagenet-multiGPU.torch/master/models/vggbn.lua
local modelType = 'D' -- on a titan black, B/D/E run out of memory even for batch-size 32

-- Create tables describing VGG configurations A, B, D, E
local cfg = {}
if modelType == 'A' then
  cfg = {64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
elseif modelType == 'B' then
  cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
elseif modelType == 'D' then
  cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 1024}
elseif modelType == 'E' then
  cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M',1024}
else
  error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
end

local features = nn.Sequential()
do
  local iChannels = 3;
  for k,v in ipairs(cfg) do
    if v == 'M' then
      features:add(nn.SpatialMaxPooling(2,2,2,2))
      features:add(nn.SpatialBatchNormalization(iChannels, 1e-3))

    else
      local oChannels = v;
      local small = math.min(iChannels, oChannels)
      local conv1a = nn.SpatialConvolution(iChannels,small,1,1)
      local conv3 = nn.SpatialConvolution(small,oChannels,3,3,1,1,1,1)
      features:add(conv1a)
      features:add(nn.ReLU(true))
      features:add(conv3)
      features:add(nn.ReLU(true))

      iChannels = oChannels;
    end
  end
end


local classifier = nn.Sequential()
classifier:add(nn.SpatialAveragePooling(7,7,1,1))
classifier:add(nn.View(1024):setNumInputDims(3))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(1024, 1000))
classifier:add(nn.LogSoftMax())

local model = nn.Sequential()
model:add(features):add(classifier)


model.inputSize = 224
model.reshapeSize = 256
model.inputMean = 128
model.inputStd = 128

return model
