mininet = {}
function mininet:new(layerTable, learingRate)
	local layers = {}
	layers[1] = layer:new()
	for i = 1,layerTable[1] do
		layers[1].nodes[i] = node:new(0)
	end
	for layerIndex = 2, #layerTable do
		layers[layerIndex] = layer:new()
		for nodeIndex = 1,layerTable[layerIndex] do
			--set number of synapses in each node to the number of nodes in previous layer
			layers[layerIndex].nodes[nodeIndex] = node:new(layerTable[layerIndex - 1])
		end
	end

	local newNN = {layers = layers, learingRate = learingRate}
	self.__index = self
	setmetatable(newNN,self)
	return newNN
end
function mininet:backPropagate(batch, targets)
	-- forward propagation
	for batchIndex = 1, #batch do
		inputs = batch[batchIndex]
		-- setup inputs in input layer
		for i = 1, #inputs do
			self.layers[1].nodes[i].values[batchIndex] = inputs[i]
		end
		-- loop through each layer
		for layerIndex = 2, #self.layers do
			currLayer = self.layers[layerIndex].nodes
			-- loop through each node in each layer
			for nodeIndex = 1, #currLayer do
				currNode = currLayer[nodeIndex]
				sum = 0
				-- loop through each input node for each node for each layer
				-- number of input nodes is equal to number of nodes in previous layer
				inputNodes = self.layers[layerIndex - 1].nodes
				for i = 1, #inputNodes do
					sum = sum + inputNodes[i].values[batchIndex] * currNode.synapses[i]
				end
				currNode.values[batchIndex] = sigmoid(sum)
			end
		end
	end

	for batchIndex = 1, #batch do
		-- backpropogation
		-- first calculate the delta for each node in outputLayer
		outputLayer = self.layers[#self.layers].nodes
		for i = 1, #outputLayer do
			guess = outputLayer[i].values[batchIndex]
			error = targets[batchIndex][i] - guess
			confidence = derivativeOfSigmoid(guess)
			outputLayer[i].deltas[batchIndex] = error * confidence
		end

		-- loop through each layer backwards, starting with first hidden layer
		for layerIndex = #self.layers - 1, 2, -1 do
			currLayer = self.layers[layerIndex].nodes
			-- prevLayer is the layer to the right
			prevLayer = self.layers[layerIndex + 1].nodes
			-- loop through each node of currLayer
			for nodeIndex = 1, #currLayer do
				currNode = currLayer[nodeIndex]
				-- calculate the delta values for nodes in currLayer by propogating back the deltas from the layer to the right (prevLayer)
				-- to do this, we must loop through nodes in layer to right, and multiply their deltas by the synapse that connects to our current node
				error = 0
				for i = 1, #prevLayer do
					bpInputNode = prevLayer[i]
					error = error + bpInputNode.deltas[batchIndex] * bpInputNode.synapses[nodeIndex]
					-- now cache the synapse weight adjustments to be applied after calculating the net adjustment for the entire batch
					bpInputNode.synapseAdjustments[nodeIndex] = bpInputNode.synapseAdjustments[nodeIndex] + currNode.values[batchIndex] * bpInputNode.deltas[batchIndex] * self.learingRate
				end
				-- to find delta, multiply error by confidence (confidence is the derivative of the value of the current node)
				currNode.deltas[batchIndex] = error * derivativeOfSigmoid(currNode.values[batchIndex])
			end
		end
		-- now adjust synapse weights between inputLayer and first hidden layer
		for nodeIndex = 1, #self.layers[2] do
			currNode = self.layers[2].nodes[nodeIndex]
			for synIndex = 1, #currNode.synapses do
				currNode.synapseAdjustments[synIndex] = currNode.synapseAdjustments[synIndex] + currNode.deltas[batchIndex] * self.layers[1].nodes[synIndex].values[batchIndex] * self.learingRate
			end
		end
	end
	-- update synapse weights using synapseAdjustments cache
	-- loop through layers
	for layerIndex = 2, #self.layers do
		-- loop through nodes
		for nodeIndex = 1, #self.layers[layerIndex].nodes do
			-- loop through synapses
			currNode = self.layers[layerIndex].nodes[nodeIndex]
			for synIndex = 1, #currNode.synapses do
				currNode.synapses[synIndex] = currNode.synapses[synIndex] + currNode.synapseAdjustments[synIndex]
				currNode.synapseAdjustments[synIndex] = 0
			end
		end
	end
end
function mininet:predictOutputs(batch)
	for batchIndex = 1, #batch do
		inputs = batch[batchIndex]
		-- setup inputs in input layer
		for i = 1, #inputs do
			self.layers[1].nodes[i].values[batchIndex] = inputs[i]
		end
		-- loop through each layer
		for layerIndex = 2, #self.layers do
			currLayer = self.layers[layerIndex].nodes
			-- loop through each node in each layer
			for nodeIndex = 1, #currLayer do
				currNode = currLayer[nodeIndex]
				sum = 0
				-- loop through each input node for each node for each layer
				-- number of input nodes is equal to number of nodes in previous layer
				inputNodes = self.layers[layerIndex - 1].nodes
				for i = 1, #inputNodes do
					--[[print("input node: ")
					print(inputNodes[i].values[1])
					print("synapse: ")
					print(currNode.synapses[i])]]
					sum = sum + inputNodes[i].values[batchIndex] * currNode.synapses[i]
				end
				currNode.values[batchIndex] = sigmoid(sum)
			end
		end
	end
	totalOutputs = {}
	for batchIndex = 1, #batch do
		outputLayer = self.layers[#self.layers]
		currBatchOut = {}
		for nodeIndex = 1, #outputLayer.nodes do
			currBatchOut[nodeIndex] = outputLayer.nodes[nodeIndex].values[batchIndex]
		end
		totalOutputs[batchIndex] = currBatchOut
	end
	return totalOutputs
end
function mininet:save(fileName)
	-- print each synapse
	local file, e = io.open(fileName, "w")
	if not file then return error(e) end
	file:write(self.learingRate .. "\n")
	for layerIndex = 1, #self.layers do
		file:write(#self.layers[layerIndex].nodes .. "-")
	end
	-- fix extra '-' because idk
	file:write("\n")
	for layerIndex = 2, #self.layers do
		-- loop through nodes
		for nodeIndex = 1, #self.layers[layerIndex].nodes do
			-- loop through synapses
			currNode = self.layers[layerIndex].nodes[nodeIndex]
			for synIndex = 1, #currNode.synapses do
				file:write(currNode.synapses[synIndex] .. "\n")
			end
		end
	end
	file:close()
end
function mininet:load(fileName)
	local layers = {}
	local data = {}
	local lineNum = 1
	for line in io.lines(fileName) do
		data[lineNum] = line
		lineNum = lineNum + 1
	end
	local learingRate = data[1]
	i = 1
	for layerSize in data[2]:gmatch("%d+") do 
		layers[i] = layer:new(layerSize)
		i = i + 1
	end
	local synIndex = 3
	for i = 1,layers[1].size do
		layers[1].nodes[i] = node:new(0)
	end
	for layerIndex = 2, #layers do
		local currLayer = layers[layerIndex]
		local numOfSynapses = layers[layerIndex - 1].size
		-- loop through nodes
		for nodeIndex = 1, currLayer.size do
			endIndex = synIndex + numOfSynapses - 1
			currLayer.nodes[nodeIndex] = node:new(numOfSynapses, {unpack(data,synIndex,endIndex)})
			synIndex = endIndex + 1
		end
	end
	local newNN = {layers = layers, learingRate = learingRate}
	self.__index = self
	return setmetatable(newNN, self)
end

layer = {}
function layer:new(size)
	-- size is just used when loading an nn from a file
	local newLayer = {nodes = {}, size = size}
	self.__index = self
	setmetatable(newLayer, self)
	return newLayer
end

node = {}
function node:new(numOfSynapses, synapseVals)
	local synapses = {}
	local synapseAdjustments = {}
	if synapseVals == nil then
		for i = 1,numOfSynapses do
			synapses[i] = 2 * math.random() - 1
			synapseAdjustments[i] = 0
		end
	else
		for i = 1,#synapseVals do
			synapses[i] = synapseVals[i]
			synapseAdjustments[i] = 0
		end
	end
	local newNode = {synapses = synapses, values = {}, deltas = {}, synapseAdjustments = synapseAdjustments}
	self.__index = self
	setmetatable(newNode, self)
	return newNode
end

-- math functions
function sigmoid(x)
	return 1 / (1 + math.exp(-x))
end
function derivativeOfSigmoid(x)
	return x * (1 - x)
end
