require 'xlua'
require 'optim'
require 'pl'
require 'trepl'
require 'nn'
require 'data'
require 'utils.log'
require 'MDReLU'
local tnt = require 'torchnet'
----------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Evaluating a convolutional network for visual classification')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model And Training Regime')
cmd:option('-batchSize',          128,                    'batch size')
cmd:option('-topK',               5,                      'measure top k error')


cmd:text('===>Platform Optimization')
cmd:option('-threads',            8,                      'number of threads')
cmd:option('-type',               'cuda',                 'cuda/cl/float/double')
cmd:option('-devid',              1,                      'device ID (if using CUDA)')
cmd:option('-nGPU',               1,                      'num of gpu devices used')


cmd:text('===>Save/Load Options')
cmd:option('-load',               '',                     'load existing net')

cmd:text('===>Data Options')
cmd:option('-dataset',            'ImageNet',              'Dataset - Cifar10, Cifar100, STL10, SVHN, MNIST')



opt = cmd:parse(arg or {})
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

local classTopK = table.prune({1, opt.topK})

----------------------------------------------------------------------
-- Model + Loss:
local model = torch.load(opt.load)
local criterion = nn.ClassNLLCriterion()
if not (model.inputMean and model.inputStd) then
  model.inputMean, model.inputStd = 118.380948, 61.896913--estimateMeanStd(trainData)
end
model.inputSize = model.inputSize or 224
model.reshapeSize = model.reshapeSize or 256

-- Data preparation
local testData = getDataset(opt.dataset, 'test')
-- classes
local classes = testData:classes()

local evalTransform = tnt.transform.compose{
  Scale(model.reshapeSize),
  CenterCrop(model.inputSize),
  Normalize(model.inputMean, model.inputStd)
}


testData = tnt.TransformDataset{
  transforms = {
    input = evalTransform
  },
  dataset = testData
}


local testIter = getIterator(testData:batch(opt.batchSize), opt.threads)


----------------------------------------------------------------------
-- Model optimization

local types = {
  cuda = 'torch.CudaTensor',
  float = 'torch.FloatTensor',
  cl = 'torch.ClTensor',
  double = 'torch.DoubleTensor'
}

local tensorType = types[opt.type] or 'torch.FloatTensor'

if opt.type == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.devid)
  local cudnnAvailable = pcall(require , 'cudnn')
  if cudnnAvailable then
    cudnn.benchmark = true
    model = cudnn.convert(model, cudnn)
  end
elseif opt.type == 'cl' then
  require 'cltorch'
  require 'clnn'
  cltorch.setDevice(opt.devid)
end

model:type(tensorType)
criterion = criterion:type(tensorType)



---Support for multiple GPUs - currently data parallel scheme
if opt.nGPU > 1 then
  local net = model
  model = nn.DataParallelTable(1)
  for i = 1, opt.nGPU do
    cutorch.setDevice(i)
    model:add(net:clone():cuda(), i)  -- Use the ith GPU
  end
  cutorch.setDevice(opt.devid)
end
-------------------------------------------------------------
print '==> Network'
print(model)

----------------------------------------------------------------------


local function forward(dataIterator, train, logName)
  local logName = logName or (train and 'Train' or 'Test')
  local yt = torch.Tensor():type(tensorType)
  local x = torch.Tensor():type(tensorType)
  local sizeData = dataIterator:execSingle('fullSize')
  local numSamples = 0
  local lossMeter = tnt.AverageValueMeter()
  local classMeter = tnt.ClassErrorMeter{topk = classTopK}
  lossMeter:reset(); classMeter:reset()

  for sample in dataIterator() do
    x:resize(sample.input:size()):copy(sample.input)
    yt:resize(sample.target:squeeze():size()):copy(sample.target)
    local y = model:forward(x)
    local loss = criterion:forward(y,yt)
    print(x:mean())
    if torch.type(y) == 'table' then y = y[1] end
    classMeter:add(y, yt)
    lossMeter:add(loss)
    numSamples = numSamples + x:size(1)
    xlua.progress(numSamples, sizeData)
  end
  return lossMeter:value(), classMeter:value()
end



local function test(dataIterator)
  model:evaluate()
  return forward(dataIterator, false)
end
------------------------------


--Test
print('Test:')
local testLoss, testClassError = test(testIter)
print(logValues('Test', epoch, testLoss, testClassError))
