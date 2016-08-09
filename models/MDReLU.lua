
-- Consider the ReLU function as g(x) and the multiplicative Gaussian 
-- drop out ReLU function as f(x).
-- for:

local MDReLU, parent = torch.class('nn.MDReLU','nn.Module') -- Modules contain two states variables: output and gradInput

function MDReLU:__init(nOutputPlane)
   parent.__init(self)

   -- we want to preform ReLU, the ReLU model uses threshold
   --  model with Parent.__init(self,0,0,p)threshold
   self.threshold = 0 or 1e-6
   self.val = 0 or 0
   
   self.train = true
   self.inplace = false
   
   -- if no argument provided, use shared model (weight is scalar)
   self.nOutputPlane = nOutputPlane or 0
   self.weight = torch.Tensor(nOutputPlane or 1):fill(0.25)
   self.gradWeight = torch.Tensor(nOutputPlane or 1)
   self.first = true
   
  -- version 2 scales output during training instead of evaluation
   --self.v2 = not v1
   
   -- for the multiplicative drop-out
   self.noise = torch.Tensor()
   self.normal_noise = torch.Tensor()
end

function printDimDebug(tensor, str)
	local dim = tensor:dim()
	local s = str..'- '
	
	for i = 1, dim, 1 do
		s = s..'x'..tensor:size(i)
	end
	print(s)
end
-- After a forward(), the output state variable should have been updated to the new value
-- It is not advised to override this function.
-- Instead, one should implement updateOutput(input) function.
-- The forward(input) function in the abstract parent class Module will call updateOutput(input).

-- Computes the output using the current parameter set of the class and input.
-- This function returns the result which is stored in the output field
function MDReLU:updateOutput(input)
   -- calculating g(x) = ReLU(x)
	if self.first then
		print("ANDREY")
		self.nOutputPlane = input:size(2)
		self.weight = torch.Tensor(self.nOutputPlane or 1):fill(0.25):cuda()
		self.noise = torch.Tensor(self.nOutputPlane or 1):cuda()
	    self.normal_noise = torch.Tensor(self.nOutputPlane or 1):cuda()
		self.gradWeight = torch.Tensor(self.nOutputPlane or 1):cuda()
		self.first = false
		
	end
   
   self.output:resize(self.weight:size(1))

   input.THNN.Threshold_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.threshold,
      self.val,
      self.inplace
   )
   

   -- f(x) = g(x) * noise  
   -- noise = \gamma * sigma
   if self.train then   -- need to add a version for test and one for train

	   for i = 1, self.nOutputPlane , 1 do
		  c = torch.abs(self.weight[i])
		  self.normal_noise[i]  = torch.normal( 1 , c)
		  self.noise[i] = (self.normal_noise[i] - 1)*c + 1  -- what happens if c negative? 
		  self.weight[i] = c
	   end
	   --[[
	   nbatch = input:size(1)
	   for i = 1 , nbatch , 1 do
		  self.output[i]:cmul(self.noise)
	   end
	   --]]
	   self.output:cmul(self.noise:view(1, self.output:size(2)):expandAs(self.output))
   end
   
   return self.output
end

-- Computing the gradient of the module with respect to its own input.
-- This is returned in gradInput.
-- Also, the gradInput state variable is updated accordingly.

function MDReLU:updateGradInput(input, gradOutput)
   
  
	--[[ 
		This computing the gradient according to the input
		and for our case it just to compute ReLU(X) gradient
	--]]
   input.THNN.Threshold_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.threshold,
      self.inplace
   )
   
   --[[
		simply mask the gradients with the noise vector.
		Since the gradient according to x is \hat{\gamma} * g'(x)
		where g(x) = ReLU(x)
	--]]
	
   if self.train then
   self.gradInput:cmul(self.noise:view(1, self.gradInput:size(2)):expandAs(self.gradInput))
   --[[
	   nbatch = input:size(1)
	   for i = 1 , nbatch , 1 do
		  self.gradInput[i]:cmul(self.noise)
	   end
	--]]
   end

   -- I'm not sure here, but it looks good from here http://stackoverflow.com/questions/36440826/how-to-write-the-updategradinput-and-accgradparameters-in-torch
   -- self.gradOutput:cmul(self.noise)
   return self.gradInput
end

-- When defining a new module, this method may need to be overloaded, if the module has trainable parameters.
-- Computing the gradient of the module with respect to its own parameters.
-- Many modules do not perform this step as they do not have any parameters.
-- The state variable name for the parameters is module dependent.
-- The module is expected to accumulate the gradients with respect to the parameters in some variable.

function MDReLU:accGradParameters(input, gradOutput, scale)
   if self.train then
	   nbatch = input:size(1)
	   self.gradWeight = torch.cmul(gradOutput[1], input[1])  --consider addr: 
	   for i = 2 , nbatch , 1 do
		  --print ( gradOutput[i]:size(1)..'x'..input[i]:size(1))
		  self.gradWeight = self.gradWeight + torch.cmul(gradOutput[i], input[i])
	   end
	   self.gradWeight = torch.cmul(self.gradWeight , (self.normal_noise - 1 )) * (scale/nbatch)
   end
	
   return self.gradWeight
end

function MDReLU:clearState()
   nn.utils.clear(self, 'gradWeightBuf', 'gradWeightBuf2')
   return parent.clearState(self)
end