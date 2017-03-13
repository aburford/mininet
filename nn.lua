nn = {}
function nn:new(layerTable, learingRate)
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
function nn:propagate(batch, targets)
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
			end
		end
	end
end

layer = {}
function layer:new()
	local newLayer = {nodes = {}}
	self.__index = self
	setmetatable(newLayer, self)
	return newLayer
end

node = {}
function node:new(numOfSynapses)
	local synapses = {}
	local synapseAdjustments = {}
	for i = 1,numOfSynapses do
		synapses[i] = 2 * math.random() - 1
		synapseAdjustments[i] = 0
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

time = os.time()
print(time)
math.randomseed(time)
myNetwork = nn:new({3,4,1},5)
for iteration = 1,1000 do
	myNetwork:propagate({{0,0,1},{1,0,1},{0,1,1},{1,1,1}},{{0},{1},{1},{0}})
end
print("00 | " .. myNetwork.layers[3].nodes[1].values[1])
print("10 | " .. myNetwork.layers[3].nodes[1].values[2])
print("01 | " .. myNetwork.layers[3].nodes[1].values[3])
print("11 | " .. myNetwork.layers[3].nodes[1].values[4])


