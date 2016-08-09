--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The full pre-activation ResNet variation from the technical report
-- "Identity Mappings in Deep Residual Networks" (http://arxiv.org/abs/1603.05027)
--
require 'nn'
require 'nngraph'
require 'AdaptiveCompute'
require 'dpnn'
local SpatialAdaptiveAveragePooling, parent = torch.class('nn.SpatialAdaptiveAveragePooling', 'nn.SpatialAveragePooling')

function SpatialAdaptiveAveragePooling:__init(W, H)
   parent.__init(self, 2, 2)

   self.W = W
   self.H = H
end

function SpatialAdaptiveAveragePooling:updateOutput(input)
    local kW = math.floor(input:size(4)/self.W)
    local kH = math.floor(input:size(3)/self.H)
    if self.kW ~= kW or self.kH ~= kH then
      parent.__init(self, kW, kH, kW, kH)
      self:type(input:type())
    end
    self.output = parent.updateOutput(self, input)
   return self.output
end

local opt = opt or {dataset='Cifar10', depth = 44, shortcutType='B'}
local Convolution = cudnn.SpatialConvolution
local AvgAll = nn.SpatialAdaptiveAveragePooling
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local AC = nn.AdaptiveCompute
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local BatchNorm = nn.BatchNormalization

local shortcutType = opt.shortcutType or 'B'
local depth = opt.depth
local iChannels


-- The shortcut layer is either identity or 1x1 convolution
local function shortcut(nInputPlane, nOutputPlane, stride)
  local useConv = shortcutType == 'C' or
  (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
  if useConv then
    -- 1x1 convolution
    return nn.Sequential()
    :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
  elseif nInputPlane ~= nOutputPlane then
    -- Strided, zero-padded identity shortcut
    return nn.Sequential()
    :add(nn.SpatialAveragePooling(1, 1, stride, stride))
    :add(nn.Concat(2)
    :add(nn.Identity())
    :add(nn.MulConstant(0)))
  else
    return nn.Identity()
  end
end



-- The basic residual layer block for 18 and 34 layer network, and the
-- CIFAR networks
local function basicblock(n, stride, type)
  local nInputPlane = iChannels
  iChannels = n

  local block = nn.Sequential()
  local s = nn.Sequential()
  if type == 'both_preact' then
    block:add(SBatchNorm(nInputPlane))
    block:add(ReLU(true))
  elseif type ~= 'no_preact' then
    s:add(SBatchNorm(nInputPlane))
    s:add(ReLU(true))
  end
  s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
  s:add(SBatchNorm(n))
  s:add(ReLU(true))
  s:add(Convolution(n,n,3,3,1,1,1,1))

  return block
  :add(nn.ConcatTable()
  :add(s)
  :add(shortcut(nInputPlane, n, stride)))
  :add(nn.CAddTable(true))
end

-- The bottleneck residual layer for 50, 101, and 152 layer networks
local function bottleneck(n, stride, type)
  local nInputPlane = iChannels
  iChannels = n * 4

  local block = nn.Sequential()
  local s = nn.Sequential()
  if type == 'both_preact' then
    block:add(SBatchNorm(nInputPlane))
    block:add(ReLU(true))
  elseif type ~= 'no_preact' then
    s:add(SBatchNorm(nInputPlane))
    s:add(ReLU(true))
  end
  s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
  s:add(SBatchNorm(n))
  s:add(ReLU(true))
  s:add(Convolution(n,n,3,3,stride,stride,1,1))
  s:add(SBatchNorm(n))
  s:add(ReLU(true))
  s:add(Convolution(n,n*4,1,1,1,1,0,0))

  return block
  :add(nn.ConcatTable()
  :add(s)
  :add(shortcut(nInputPlane, n * 4, stride)))
  :add(nn.CAddTable(true))
end

-- Creates count residual blocks with specified number of features
local function layer(input, block, features, count, stride, type, outputs)
  if count < 1 then
    return input
  end
  local out = block(features, stride,
  type == 'first' and 'no_preact' or 'both_preact')(input)
  table.insert(outputs, outLayer(features, 64)(out))

  for i=2,count do
    out = block(features, 1)(out)
    table.insert(outputs, outLayer(features, 64)(out))

  end
  return out
end

local hUnitShared = nn.Linear(64,1)
hUnitShared.bias:fill(-5)
function outLayer(nIn, nOut)
  local m = nn.Sequential()
  m:add(SBatchNorm(nIn))
  m:add(AvgAll(1,1))
  m:add(nn.View(-1):setNumInputDims(3))
  m:add(nn.Linear(nIn, nOut))
  m:add(BatchNorm(nOut, nil,nil,false))

  local cat = nn.ConcatTable()
  cat:add(nn.Identity())
  cat:add(nn.Sequential():add(hUnitShared:clone('weight','gradWeight','bias','gradBias')):add(nn.Sigmoid()))
  -- local cat = nn.ConcatTable()
  -- local hUnit = nn.Linear(nIn,1)
  -- hUnit.bias:fill(-5)
  -- cat:add(nn.Sequential():add(nn.Linear(nIn, nOut)))
  -- cat:add(nn.Sequential():add(hUnit):add(nn.Sigmoid()))

  m:add(cat)
  return m
end


local model = nn.Sequential()
if opt.dataset == 'ImageNet' then
  -- Configurations for ResNet:
  --  num. residual blocks, num features, residual block function
  local cfg = {
    [18]  = {{2, 2, 2, 2}, 512, basicblock},
    [34]  = {{3, 4, 6, 3}, 512, basicblock},
    [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
    [101] = {{3, 4, 23, 3}, 2048, bottleneck},
    [152] = {{3, 8, 36, 3}, 2048, bottleneck},
    [200] = {{3, 24, 36, 3}, 2048, bottleneck},
  }

  assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
  local def, nFeatures, block = table.unpack(cfg[depth])
  iChannels = 64
  print(' | ResNet-' .. depth .. ' ImageNet')

  -- The ResNet ImageNet model
  model:add(Convolution(3,64,7,7,2,2,3,3))
  model:add(SBatchNorm(64))
  model:add(ReLU(true))
  model:add(Max(3,3,2,2,1,1))
  model:add(layer(block, 64,  def[1], 1, 'first'))
  model:add(layer(block, 128, def[2], 2))
  model:add(layer(block, 256, def[3], 2))
  model:add(layer(block, 512, def[4], 2))
  model:add(SBatchNorm(iChannels))
  model:add(ReLU(true))
  model:add(Avg(7, 7, 1, 1))
  model:add(nn.View(nFeatures):setNumInputDims(3))
  model:add(nn.Linear(nFeatures, 1000))
elseif opt.dataset == 'Cifar10' then
  -- Model type specifies number of layers for CIFAR-10 model
  depth = 44
  assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
  local n = (depth - 2) / 6
  iChannels = 16
  print(' | ResNet-' .. depth .. ' CIFAR-10')

  local input = nn.Identity()()
  currOutput = input
  local outputs = {}
  -- The ResNet CIFAR-10 model
  currOutput = Convolution(3,16,3,3,1,1,1,1)(currOutput)
  currOutput = layer(currOutput, basicblock, 16, n, 1, nil, outputs)
  currOutput = layer(currOutput, basicblock, 32, n, 2, nil, outputs)
  currOutput = layer(currOutput, basicblock, 64, n, 2, nil, outputs)
  --
  -- local weighted = {}
  -- for i,l in pairs(outputs) do
  --   local sig, feat = l:split(2)
  --   table.insert(weighted, nn.CMulTable()({nn.Squeeze()(nn.Replicate(64,2)(sig)), feat}))
  -- end
  local classifier = nn.Sequential()
  classifier:add(BatchNorm(64))
  classifier:add(nn.ReLU())
  classifier:add(nn.Linear(64,10))
  classifier:add(nn.LogSoftMax())
  model = nn.gModule({input}, {classifier(AC(0.01, 1e-3)(outputs))})
  --
  -- currOutput = SBatchNorm(iChannels)(currOutput)
  -- currOutput = ReLU(true)(currOutput)
  -- currOutput = Avg(8, 8, 1, 1)(currOutput)
  -- currOutput = nn.View(64):setNumInputDims(3)(currOutput)
  -- currOutput = nn.Linear(64, 10)(currOutput)
else
  error('invalid dataset: ' .. opt.dataset)
end
--
-- --model:add(nn.LogSoftMax())

local function ConvInit(name)
  for k,v in pairs(model:findModules(name)) do
    local n = v.kW*v.kH*v.nOutputPlane
    v.weight:normal(0,math.sqrt(2/n))
    v:noBias()
  end
end
local function BNInit(name)
  for k,v in pairs(model:findModules(name)) do
    v.weight:fill(1)
    v.bias:zero()
  end
end

ConvInit('nn.SpatialConvolution')
ConvInit('nn.SpatialConvolution')
BNInit('nn.SpatialBatchNormalization')
BNInit('nn.SpatialBatchNormalization')
BNInit('nn.SpatialBatchNormalization')
-- for k,v in pairs(model:findModules('nn.Linear')) do
--   v.bias:zero()
-- end
--
--
-- model:get(1).gradInput = nil

return model
