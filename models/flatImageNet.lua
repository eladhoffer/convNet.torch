require 'nn'

local model = nn.Sequential()
model:add(nn.View(-1):setNumInputDims(3))
model:add(nn.Linear(3*64*64,105))
model:add(nn.ReLU())
model:add(nn.Linear(105,1))
--model:add(nn.Tanh())

return model
