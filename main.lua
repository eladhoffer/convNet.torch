require 'xlua'
require 'optim'
require 'pl'
require 'trepl'
require 'nn'
require 'data'
require 'utils.log'
local tnt = require 'torchnet'
----------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a convolutional network for visual classification')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model And Training Regime')
cmd:option('-modelsFolder',      './models/',             'Models Folder')
cmd:option('-model',             'model.lua',             'Model file - must return valid network.')
cmd:option('-LR',                 0.1,                    'learning rate')
cmd:option('-LRDecay',            0,                      'learning rate decay (in # samples)')
cmd:option('-weightDecay',        1e-4,                   'L2 penalty on the weights')
cmd:option('-momentum',           0.9,                    'momentum')
cmd:option('-batchSize',          128,                    'batch size')
cmd:option('-optimization',       'sgd',                  'optimization method')
cmd:option('-epoch',              -1,                     'number of epochs to train, -1 for unbounded')
cmd:option('-evalN',              100000,                 'evaluate every N samples')
cmd:option('-topK',               1,                      'measure top k error')


cmd:text('===>Platform Optimization')
cmd:option('-threads',            8,                      'number of threads')
cmd:option('-type',               'cuda',                 'cuda/cl/float/double')
cmd:option('-devid',              1,                      'device ID (if using CUDA)')
cmd:option('-nGPU',               1,                      'num of gpu devices used')
cmd:option('-saveOptState',       false,                  'Save optimization state every epoch')


cmd:text('===>Save/Load Options')
cmd:option('-load',               '',                     'load existing net')
cmd:option('-resume',             false,                  'resume training from the same epoch')
cmd:option('-save',               os.date():gsub(' ',''), 'save directory')

cmd:text('===>Data Options')
cmd:option('-dataset',            'Cifar10',              'Dataset - Cifar10, Cifar100, STL10, SVHN, MNIST')
cmd:option('-normalization',      'simple',               'simple - whole sample, channel - by image channel, image - mean and std images')
cmd:option('-format',             'rgb',                  'rgb or yuv')
cmd:option('-whiten',             false,                  'whiten data')
cmd:option('-augment',            true,                  'Augment training data')
cmd:option('-preProcDir',         './preProcData/',       'Data for pre-processing (means,P,invP)')

cmd:text('===>Misc')
cmd:option('-showPlot',           true,                   'Display plot each epoch')


opt = cmd:parse(arg or {})
opt.model = opt.modelsFolder .. paths.basename(opt.model, '.lua')
opt.save = paths.concat('./results', opt.save)
opt.preProcDir = paths.concat(opt.preProcDir, opt.dataset .. '/')
os.execute('mkdir -p ' .. opt.preProcDir)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

-- Output files configuration
os.execute('mkdir -p ' .. opt.save)
cmd:log(opt.save .. '/Log.txt', opt)
local logFile = paths.concat(opt.save, 'LogTable.csv')
local classTopK = table.prune({1, opt.topK})
local log = getLog(logFile, {'Train', 'Test'}, classTopK)
local netFilename = paths.concat(opt.save, 'savedModel')

----------------------------------------------------------------------
-- Model + Loss:
local model, criterion
if paths.filep(opt.load) then
    local conf, criterion = require(opt.model)
    model = torch.load(opt.load)
    if not opt.resume then
      for _, val in pairs({'epoch', 'regime', 'optimState'}) do
        model[val] = conf[val]
      end
    end
else
    model, criterion = require(opt.model)
end

criterion = criterion or nn.ClassNLLCriterion()

if not (model.inputMean and model.inputStd) then
  model.inputMean, model.inputStd = 128, 128--estimateMeanStd(trainData)
end
model.inputSize = model.inputSize or 32
model.reshapeSize = model.reshapeSize or model.inputSize

-- Data preparation
local trainData = getDataset(opt.dataset, 'train')
local testData = getDataset(opt.dataset, 'test')
-- classes
local classes = trainData:classes()

local evalTransform = tnt.transform.compose{
  Scale(model.reshapeSize),
  CenterCrop(model.inputSize),
  Normalize(model.inputMean, model.inputStd)
}

local augTransform = tnt.transform.compose{
  --RandomScale(model.inputSize, model.reshapeSize * 1.2),
  --Scale(model.reshapeSize),
  RandomCrop(model.inputSize, 4),
  HorizontalFlip(),
  Normalize(model.inputMean,model.inputStd),
--  ColorJitter(0.4,0.4,0.4)
}

testData = tnt.TransformDataset{
    transforms = {
      input = evalTransform
    },
    dataset = testData
}

trainData =  tnt.TransformDataset{
    transforms = {
      input = (opt.augment and augTransform) or evalTransform
    },
    dataset = trainData:shuffle()
}

local trainIter = getIterator(trainData:batch(opt.batchSize), opt.threads)
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

-- local optnetAvailable, optnet = pcall(require , 'optnet')
-- if optnetAvailable and opt.type == 'cuda' then
--   local input = trainData:get(1)
--   optnet.optimizeMemory(model, input, {mode='training'})
-- end
-- Optimization configuration
local weights,gradients = model:getParameters()
local savedModel = model
if opt.nGPU > 1 then
  savedModel = savedModel:get(1)
end
savedModel = savedModel:clone('weight', 'bias', 'running_mean', 'running_std', 'running_var')

------------------Optimization Configuration--------------------------
local optimState = model.optimState or {
    method = opt.optimization,
    learningRate = opt.LR,
    momentum = opt.momentum,
    dampening = 0,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay
}

----------------------------------------------------------------------
print '==> Network'
print(model)
print('==>' .. weights:nElement() ..  ' Parameters')

print '==> Criterion'
print(criterion)

----------------------------------------------------------------------
if opt.saveOptState then
  model.optimState = optimState
end

local epoch = (model.epoch and model.epoch + 1) or 1


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
        if train then
            local function feval()
                model:zeroGradParameters()
                local dE_dy = criterion:backward(y, yt)
                model:backward(x, dE_dy)
                return loss, gradients
            end
            _G.optim[optimState.method](feval, weights, optimState)
            if opt.nGPU > 1 then
                model:syncParameters()
            end
        end
        if torch.type(y) == 'table' then y = y[1] end
        classMeter:add(y, yt)
        lossMeter:add(loss, x:size(1) / opt.batchSize)
        numSamples = numSamples + x:size(1)
        xlua.progress(numSamples, sizeData)
        if numSamples % opt.evalN < opt.batchSize then
          print(logValues('Train', epoch - 1 + numSamples / sizeData, lossMeter:value(), classMeter:value()))
        end
    end
    return lossMeter:value(), classMeter:value()
end

------------------------------
local function train(dataIterator)
    model:training()
    return forward(dataIterator, true)
end

local function test(dataIterator)
    model:evaluate()
    return forward(dataIterator, false)
end
------------------------------

local lowestTestError = 100
print '\n==> Starting Training\n'

while epoch ~= opt.epoch do
    model.epoch = epoch
    print('\nEpoch ' .. epoch)
    updateOpt(optimState, epoch, model.regime, true)
    print('Training:')
    --Train
    trainIter:exec('manualSeed', epoch)
    trainIter:exec('resample')
    local trainLoss, trainClassError = train(trainIter)
    torch.save(netFilename, savedModel)

    log:set(logValues('Train', epoch, trainLoss, trainClassError))

    --Test
    print('Test:')
    local testLoss, testClassError = test(testIter)
    log:set(logValues('Test', epoch, testLoss, testClassError))
    log:flush()

    if lowestTestError > testClassError[1] then
      lowestTestError = testClassError[1]
      os.execute(('cp %s %s'):format(netFilename, netFilename .. '_best'))
    end

    epoch = epoch + 1
end
