require 'nn'
require 'nngraph'
local SpatialConvolution = nn.SpatialConvolution
local ReLU = nn.ReLU
local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialMaxPooling = nn.SpatialMaxPooling
local SpatialAveragePooling = nn.SpatialAveragePooling

local function ConvBN(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  local module = nn.Sequential()
  module:add(SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
  module:add(SpatialBatchNormalization(nOutputPlane,1e-3,nil,true))
  module:add(ReLU(true))
  return module
end

local function Tower(tbl)
  local module = nn.Sequential()
  for i=1, #tbl do
    module:add(tbl[i])
  end
  return module
end

local function Stem()
  local module = nn.Sequential()
  module:add(ConvBN(3,32,3,3,2,2,0,0))
  module:add(ConvBN(32,32,3,3,1,1,0,0))
  module:add(ConvBN(32,64,3,3,1,1,1,1))

  local cat1 = nn.Concat(2)
  cat1:add(SpatialMaxPooling(3,3,2,2,0,0))
  cat1:add(ConvBN(64,96,3,3,2,2,0,0))

  local cat2 = nn.Concat(2)
  cat2:add(Tower{
      ConvBN(160,64,1,1,1,1,0,0),
      ConvBN(64,64,7,1,1,1,3,0),
      ConvBN(64,64,1,7,1,1,0,3),
      ConvBN(64,96,3,3,1,1,0,0)
    })
  cat2:add(Tower{
      ConvBN(160,64,1,1,1,1,0,0),
      ConvBN(64,96,3,3,1,1,0,0)
    })

  local cat3 = nn.Concat(2)
  cat3:add(SpatialMaxPooling(3,3,2,2,0,0))
  cat3:add(ConvBN(192,192,3,3,2,2,0,0))

  module:add(cat1)
  module:add(cat2)
  module:add(cat3)

  return module
end

local function Reduction_A(nInputPlane,k,l,m,n)
  local module = nn.Concat(2)
  module:add(SpatialMaxPooling(3,3,2,2,0,0))
  module:add(ConvBN(nInputPlane,n,3,3,2,2,0,0))
  module:add(Tower{
      ConvBN(nInputPlane,k,1,1,1,1,0,0),
      ConvBN(k,l,3,3,1,1,1,1),
      ConvBN(l,m,3,3,2,2,0,0)
    })
  return module
end

local function Reduction_B(nInputPlane)
  local module = nn.Concat(2)
  module:add(SpatialMaxPooling(3,3,2,2,0,0))
  module:add(Tower{
      ConvBN(nInputPlane,192,1,1,1,1,0,0),
      ConvBN(192,192,3,3,2,2,0,0)
    })
  module:add(Tower{
      ConvBN(nInputPlane,256,1,1,1,1,0,0),
      ConvBN(256,256,1,7,1,1,0,3),
      ConvBN(256,320,7,1,1,1,3,0),
      ConvBN(320,320,3,3,2,2,0,0)

    })
  return module
end

local function Inception_A(nInputPlane)
  local inceptionModule = nn.Concat(2)
  inceptionModule:add(Tower{
      SpatialAveragePooling(3,3,1,1,1,1),
      ConvBN(nInputPlane,96,1,1,1,1,0,0),
    })
  inceptionModule:add(ConvBN(nInputPlane,96,1,1,1,1,0,0))
  inceptionModule:add(Tower{
      ConvBN(nInputPlane,64,1,1,1,1,0,0),
      ConvBN(64,96,3,3,1,1,1,1)
    })
  inceptionModule:add(Tower{
      ConvBN(nInputPlane,64,1,1,1,1,0,0),
      ConvBN(64,96,3,3,1,1,1,1),
      ConvBN(96,96,3,3,1,1,1,1)
    })

  return inceptionModule
end

local function Inception_B(nInputPlane)
  local inceptionModule = nn.Concat(2)
  inceptionModule:add(Tower{
      SpatialAveragePooling(3,3,1,1,1,1),
      ConvBN(nInputPlane,128,1,1,1,1,0,0)
    })
  inceptionModule:add(ConvBN(nInputPlane,384,1,1,1,1,0,0))
  inceptionModule:add(Tower{
      ConvBN(nInputPlane,192,1,1,1,1,0,0),
      ConvBN(192,224,1,7,1,1,0,3),
      ConvBN(224,256,7,1,1,1,3,0)
    })
  inceptionModule:add(Tower{
      ConvBN(nInputPlane,192,1,1,1,1,0,0),
      ConvBN(192,192,1,7,1,1,0,3),
      ConvBN(192,224,7,1,1,1,3,0),
      ConvBN(224,224,1,7,1,1,0,3),
      ConvBN(224,256,7,1,1,1,3,0)
    })
  return inceptionModule
end

local function Inception_C(nInputPlane)
  local inceptionModule = nn.Concat(2)
  inceptionModule:add(Tower{
      SpatialAveragePooling(3,3,1,1,1,1),
      ConvBN(nInputPlane,256,1,1,1,1,0,0)
    })
  inceptionModule:add(ConvBN(nInputPlane,256,1,1,1,1,0,0))
  inceptionModule:add(Tower{
      ConvBN(nInputPlane,384,1,1,1,1,0,0),
      nn.Concat(2):add(ConvBN(384,256,3,1,1,1,1,0)):add(ConvBN(384,256,1,3,1,1,0,1))
    })
  inceptionModule:add(Tower{
      ConvBN(nInputPlane,384,1,1,1,1,0,0),
      ConvBN(384,448,1,3,1,1,0,1),
      ConvBN(448,512,3,1,1,1,1,0),
      nn.Concat(2):add(ConvBN(512,256,3,1,1,1,1,0)):add(ConvBN(512,256,1,3,1,1,0,1))
    })
  return inceptionModule
end

local auxClassifier = function(nInputDims, mapSize)
  local m = nn.Sequential()
  m:add(nn.SpatialAveragePooling(5,5,3,3):ceil())
  m:add(nn.SpatialConvolution(nInputDims,128,1,1))
  m:add(nn.ReLU(true))
  m:add(nn.View(-1):setNumInputDims(3))
  m:add(nn.Linear(128*mapSize*mapSize,768))
  m:add(nn.ReLU(true))
  m:add(nn.Dropout(0.2))
  m:add(nn.Linear(768,1000))
  m:add(nn.LogSoftMax())
  return m
end

local mainClassifier = function()
  local m = nn.Sequential()
  m:add(SpatialAveragePooling(8,8,1,1))
  m:add(nn.View(-1):setNumInputDims(3))
  m:add(nn.Dropout(0.2))
  m:add(nn.Linear(1536, 1000))
  m:add(nn.LogSoftMax())
  return m
end

local input = nn.Identity()()
local model = Stem()(input)

for i=1,4 do
  model = Inception_A(384)(model)
end
model = Reduction_A(384,192,224,256,384)(model)
local branch1 = auxClassifier(1024, 5)(model)

for i=1,7 do
  model = Inception_B(1024)(model)
end

model = Reduction_B(1024)(model)
local branch2 = auxClassifier(1536, 2)(model)

for i=1,3 do
  model = Inception_C(1536)(model)
end

local mainBranch = mainClassifier()(model)
local model = nn.gModule({input},{mainBranch,branch1,branch2})

local NLL = nn.ClassNLLCriterion()
local loss = nn.ParallelCriterion(true):add(NLL):add(NLL,0.3):add(NLL,0.3)

model.inputSize = 299
model.reshapeSize = 384
model.inputMean = 128
model.inputStd = 128
model.regime = {
  epoch = {1, 15, 25, 30, 35},
  learningRate = {1e-2, 5e-3, 1e-3, 5e-4, 1e-4},
  weightDecay = {5e-4, 5e-4, 0 }
}
return model, loss
