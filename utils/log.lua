local tnt = require 'torchnet'
local logtext = require 'torchnet.log.view.text'
local logstatus = require 'torchnet.log.view.status'
local _, display = pcall(require , 'display')

local plotValues = {}
local function addPlot(name, keys, ylabel)
  if not display then return end

  local ylabel = ylabel or 'value'
  plotValues[name] = {}
  plotValues[name].config = {
    title = name,
    labels = keys,
    ylabel = ylabel
  }
  plotValues[name].data = {}
  return function(log)
    local config = plotValues[name].config
    local entry = {}
    for _, key in ipairs(keys) do
      local val = log:get{key = key}
      table.insert(entry, val)
    end
    table.insert(plotValues[name].data, entry)
    config.win = display.plot(plotValues[name].data, config)

  end
end

function logValues(name, epoch, loss, error)
  local values = {Epoch = epoch}
  values[name .. ' Loss'] = loss
  for k,val in pairs(error) do
    values[('%s Error (Top %s)'):format(name, k)] = val
  end
  return values
end

function getLog(logFile, entries, classTopK)
  local classTopK = classTopK or {1,5}

  local logKeys = {'Epoch'}
  local logKeysFormat = {'%5.3f'}
  local logFormat = {'%5s'}
  for _, e in pairs(entries) do
    table.insert(logKeys, ('%s Loss'):format(e))
    table.insert(logKeysFormat, '%10.3f')
    table.insert(logFormat, '%10s')
    for _, k in pairs(classTopK) do
      table.insert(logKeys, ('%s Error (Top %s)'):format(e,k))
      table.insert(logKeysFormat, '%10.3f')
      table.insert(logFormat, '%10s')
    end
  end


  local log = tnt.Log{
    keys = logKeys,
    onFlush = {
      -- write out all keys in "log" file
      logtext{filename=logFile, keys=logKeys, format=logKeysFormat},
      logtext{keys=logKeys},
      addPlot('Classification Error', {'Epoch','Train Error (Top 1)', 'Test Error (Top 1)'}, 'Error %'),
    },
    onSet = {
      -- add status to log
      logstatus{filename=logFile}
    }
  }

  log:set{
    __status__ = string.format(table.concat(logFormat, ' | '), unpack(logKeys)),
  }
  return log
end

function updateOpt(optState, epoch, regime, verbose)
  if regime and regime.epoch then
    for epochNum, epochVal in pairs(regime.epoch) do
      if epochVal == epoch then
        for optValue,_ in pairs(regime) do
          if regime[optValue][epochNum] then
            if verbose then
              print(optValue,': ',optState[optValue], ' -> ', regime[optValue][epochNum])
            end
            optState[optValue] = regime[optValue][epochNum]
          end
        end
      end
    end
  end
end
