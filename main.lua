require 'xlua'
require 'optim'
require 'pl'
require 'trepl'
require 'nn'
require 'data'
require 'utils.log'
require 'utils.plotCSV'
local tnt = require 'torchnet'
----------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a convolutional network for visual classification')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model And Training Regime')
cmd:option('-modelsFolder', './models/', 'Models Folder')
cmd:option('-resultsFolder', './results/', 'Models Folder')
cmd:option('-model', 'model.lua', 'Model file - must return valid network.')
cmd:option('-LR', 0.1, 'learning rate')
cmd:option('-LRDecay', 0, 'learning rate decay (in # samples)')
cmd:option('-weightDecay', 1e-4, 'L2 penalty on the weights')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-batchSize', 128, 'batch size')
cmd:option('-optimization', 'sgd', 'optimization method')
cmd:option('-epoch', -1, 'number of epochs to train, -1 for unbounded')
cmd:option('-evalN', 100000, 'evaluate every N samples')
cmd:option('-topK', 5, 'measure top k error')

cmd:text('===>Platform Optimization')
cmd:option('-threads', 8, 'number of threads')
cmd:option('-type', 'cuda', 'cuda/cl/float/double')
cmd:option('-devid', 1, 'device ID (if using CUDA)')
cmd:option('-nGPU', 1, 'num of gpu devices used')
cmd:option('-saveOptState', false, 'Save optimization state every epoch')

cmd:text('===>Save/Load Options')
cmd:option('-load', '', 'load existing net')
cmd:option('-resume', false, 'resume training from the same epoch')
cmd:option('-save', os.date():gsub(' ',''), 'name of saved directory')

cmd:text('===>Data Options')
cmd:option('-dataset', 'Cifar10', 'Dataset - ImageNet, Cifar10, Cifar100, STL10, SVHN, MNIST')
cmd:option('-augment', true, 'Augment training data')

opt = cmd:parse(arg or {})
opt.model = opt.modelsFolder .. paths.basename(opt.model, '.lua')
opt.savePath = paths.concat(opt.resultsFolder, opt.save)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

-- Output files configuration
os.execute('mkdir -p ' .. opt.savePath)
cmd:log(opt.savePath .. '/Log.txt', opt)
local logFile = paths.concat(opt.savePath, 'LogTable.csv')
local classTopK = table.prune({1, opt.topK})

local log = getLog{
    logFile = logFile,
    keys = {
        [1] = 'Epoch', [2] = 'Train Loss', [3] = 'Test Loss',
        ['Train Error'] = classTopK, ['Test Error'] = classTopK
    },
    nestFormat ='%s (Top %s)'
}

local plots = {
    {
        title = opt.save:gsub('_','-') .. ':Loss',
        labels = {'Epoch', 'Train Loss', 'Test Loss'},
        ylabel = 'Loss'
    }
}

for i,k in pairs(classTopK) do
    table.insert(plots,
    {
        title = ('%s : Classification Error (Top %s)'):format(opt.save:gsub('_','-'), k),
        labels = {'Epoch', ('Train Error (Top %s)'):format(k), ('Test Error (Top %s)'):format(k)},
        ylabel = 'Error %'
    }
    )
end

log:attach('onFlush',
{
    function()
        local plot = PlotCSV(logFile)
        plot:parse()
        for _,p in pairs(plots) do
            if pcall(require , 'display') then
                p.win = plot:display(p)
            end
            plot:save(paths.concat(opt.savePath, p.title:gsub('%s','') .. '.eps'), p)
        end
    end
}
)

local netFilename = paths.concat(opt.savePath, 'savedModel')

----------------------------------------------------------------------
local config = {
    inputSize = 32,
    reshapeSize = 32,
    inputMean = 128,
    inputStd = 128,
    regime = {},
    epoch = 0
}
local function setConfig(target, origin, overwrite)
    for key in pairs(config) do
        if overwrite or target[key] == nil then
            target[key] = origin[key]
        end
    end
end

-- Model + Loss:
local model, criterion
if paths.filep(opt.load) then
    local conf
    conf, criterion = require(opt.model)
    model = torch.load(opt.load)
    inflateGradModel(model)
    if not opt.resume then
        setConfig(model, conf)
    end
else
    model, criterion = require(opt.model)
end

criterion = criterion or nn.ClassNLLCriterion()

setConfig(model, config)

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
    Scale(model.reshapeSize),
    RandomCrop(model.inputSize),
    HorizontalFlip(),
    Normalize(model.inputMean,model.inputStd),
    -- ColorJitter(0.4,0.4,0.4)
}

testData = tnt.TransformDataset{
    transforms = {
        input = evalTransform
    },
    dataset = testData
}

trainData = tnt.TransformDataset{
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

local tensorType = types[opt.type] or opt.type

if opt.type == 'cuda' then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.devid)
    local cudnnAvailable = pcall(require , 'cudnn')
    if cudnnAvailable then
        cudnn.benchmark = true
        model:type(tensorType)
        model = cudnn.convert(model, cudnn)
    end
elseif opt.type == 'cl' then
    require 'cltorch'
    require 'clnn'
    cltorch.setDevice(opt.devid)
end

criterion = criterion:type(tensorType)

model:type(tensorType)

---Support for multiple GPUs - currently data parallel scheme
if opt.nGPU > 1 then
    local net = model
    model = nn.DataParallelTable(1, true, true)
    local useCudnn = cudnn ~= nil
    local modelConf = opt.model
    model:add(net, torch.range(1, opt.nGPU):totable()) -- Use the ith GPU
    model:threads(function()
        require(modelConf)
        if useCudnn then
            require 'cudnn'
            cudnn.benchmark = true
        end
    end
    )
    setConfig(model, net, true)
end
print(model)
-- Optimization configuration
local weights,gradients = model:getParameters()
local savedModel = model
if opt.nGPU > 1 then
    savedModel = savedModel:get(1)
end
savedModel = clonedSavedModel(savedModel)

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
print('==>' .. weights:nElement() .. ' Parameters')

print '==> Criterion'
print(criterion)

print '==> Regime'
print(model.regime)
----------------------------------------------------------------------
if not opt.savePathOptState then
    model.optimState = nil
end
local epoch = (model.epoch and model.epoch + 1) or 1

local function forward(dataIterator, train)
    local yt = torch.Tensor():type(tensorType)
    local x = torch.Tensor():type(tensorType)
    local sizeData = dataIterator:execSingle('fullSize')
    local numSamples = 0
    local avgLoss = 0
    local classMeter = tnt.ClassErrorMeter{topk = classTopK}
    classMeter:reset()

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
                local norm = gradients:norm()
                if norm > 10 then
                    gradients:mul(10/norm)
                end
                return loss, gradients
            end
            _G.optim[optimState.method](feval, weights, optimState)
            if opt.nGPU > 1 then
                model:syncParameters()
            end
        end
        if torch.type(y) == 'table' then y = y[1] end
        classMeter:add(y, yt)
        avgLoss = avgLoss + loss * x:size(1)
        numSamples = numSamples + x:size(1)
        xlua.progress(numSamples, sizeData)
        if numSamples % opt.evalN < opt.batchSize then
            print('Current Loss: ' .. avgLoss / numSamples)
            print('Current Error: ' .. classMeter:value()[1])
        end
    end
    avgLoss = avgLoss / numSamples
    return avgLoss, classMeter:value()
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
    log:set{Epoch = epoch}
    print('\nEpoch ' .. epoch)
    updateOpt(optimState, epoch, model.regime, true)
    print('Training:')
    --Train
    trainIter:exec('manualSeed', epoch)
    trainIter:exec('resample')
    local trainLoss, trainClassError = train(trainIter)
    log:set{['Train Loss'] = trainLoss, ['Train Error'] = trainClassError}
    torch.save(netFilename, savedModel)
    torch.save(netFilename .. '_full_' .. epoch, model:get(1))

    --Test
    print('Test:')
    local testLoss, testClassError = test(testIter)
    log:set{['Test Loss'] = testLoss, ['Test Error'] = testClassError}

    log:flush()

    if lowestTestError > testClassError[1] then
        lowestTestError = testClassError[1]
        os.execute(('cp %s %s'):format(netFilename, netFilename .. '_best'))
    end

    epoch = epoch + 1
end
