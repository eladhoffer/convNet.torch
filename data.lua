tnt = require 'torchnet'
local argcheck = require 'argcheck'
require 'utils.transforms'
require 'utils.imageNetData'


getImageNet = argcheck{
  {name='mode', type='string',default='train'},
  {name='path', type='string', default='/media/ehoffer/SSD/Datasets/ImageNet'},
  call = function (mode, path)
    if mode ~= 'train' then
      mode = 'validation'
    end
    local db = tnt.IndexedDataset{
      fields = {mode},
      path = path,
      mmapidx = true
    }
    function getImage(x)
      return {
        input = image.decompress(x[mode].input, 3, 'byte'),
        target = torch.ByteTensor({ImageNetClassNum(x[mode].target)})
      }
    end

    db = tnt.TransformDataset{
        transform = getImage,
        dataset = db
      }

    db.classes = function(self) return ImageNetclasses end
    db.fullSize = function() return db:size() end
    db.manualSeed = function(self, num) torch.manualSeed(num) end

    return db
  end
}


getFileNames = argcheck{
  {name='path', type='string', default='/media/ehoffer/SSD/Datasets/'},
  {name='ext', type='string', default='jpg'},
  call = function (path, ext)
    local filenames = sys.execute(("find %s -iname '*.%s'"):format(path, ext)):split('\n')
    return tnt.TableDataset{data = filenames}
  end
}


getDataset = argcheck{
  {name='dataset', type='string',default='Cifar10'},
  {name='mode', type='string',default='train'},
  {name='path', type='string', default='/home/ehoffer/Datasets/'},
  {name='preProcDir', type='string', default='./'},
  call = function (dataset, mode, path, preProcDir)
    local classes
    local loadedFile
    if dataset == 'ImageNet' then
      return getImageNet{mode=mode}
    elseif dataset =='Cifar100' then
      loadedFile = torch.load(path .. 'Cifar100/cifar100-' .. mode .. '.t7')
      loadedFile.label = loadedFile.label:add(1):byte()
      classes = torch.linspace(1,100,100):storage():totable()
    elseif dataset == 'Cifar10' then
      loadedFile = torch.load(path .. 'Cifar10/cifar10-' .. mode .. '.t7')
      loadedFile.label = loadedFile.label:add(1):byte()
      classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
    elseif dataset == 'STL10' then
      loadedFile = torch.load(path .. 'STL10/stl10-' .. mode .. '.t7')

      if mode == 'unlabeled' then
        loadedFile.label = torch.ByteTensor({0}):expand(loadedFile.data:size(1))
      else
        loadedFile.label:add(1)
      end
      classes = {'airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'}
    elseif dataset == 'tinyImageNet' then
      loadedFile = torch.load(path .. 'ImageNet/tinyImageNet.t7')
    elseif dataset == 'MNIST' then
      loadedFile = torch.load(path .. 'MNIST/mnist-' .. mode .. '.t7')
      loadedFile.data = loadedFile.data:view(loadedFile.data:size(1), 1, 28, 28)
      loadedFile.label[loadedFile.label:eq(0)] = 10
      classes = {1,2,3,4,5,6,7,8,9,0}
    elseif dataset == 'SVHN' then
      loadedFile = torch.load(path .. 'SVHN/' .. mode .. '_32x32.t7','ascii')
      loadedFile.label[loadedFile.label:eq(0)] = 10
      classes = {1,2,3,4,5,6,7,8,9,0}
    end
    local db = tnt.ListDataset{
      list = torch.LongTensor():range(1,loadedFile.data:size(1)):totable(),
      load = function(idx)
        return {
          input  = loadedFile.data[idx],
          target = torch.ByteTensor({loadedFile.label[idx]})
        } -- sample contains input and target
      end
    }
    db.classes = function(self) return classes end
    db.fullSize = function() return db:size() end
    db.manualSeed = function(sef, num) torch.manualSeed(num) end

    return db
  end
}


getIterator = argcheck{
  {name='dataset', type='tnt.Dataset'},
  {name='nthread', type='number', default=8},
  call = function (dataset, nthread)
    local iter = tnt.ParallelDatasetIterator{
      nthread = nthread,
      init    = function()
        require 'data'
      end,
      closure = function()
        return dataset
      end
    }
    return iter
  end
}

function estimateMeanStd(data, numEst, typeVal)
  --assumes all data samples are of the same size
  local typeVal = typeVal or 'simple'
  local numEst = math.min(data:size(), numEst or 10000)
  local x = data:batch(numEst):get(1).input:float()
  return x:mean(), x:std()
end
