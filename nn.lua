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
function nn:propagate(batch, target)
	-- forward propagation
	for batchIndex = 1, #batch do
		inputs = batch[batchIndex]
		-- setup inputs in input layer
		for i = 1, #inputs do
			self.layers[1].nodes[i].value = inputs[i]
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
					sum  = sum + inputNodes[i].values[batchIndex] * currNode.synapses[i]
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
			error = target[i] - guess
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
					-- now adjust the synapse weights
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
				currNode.synapseAdjustments[synIndex] = currNode.synapseAdjustments[synIndex] = currNode.deltas[batchIndex] * self.layers[1].nodes[synIndex]
				currNode.synapses[synIndex] = currNode.synapses[synIndex] + currNode.delta * input * self.learingRate
			end
		end
	end
	bpInputNode.synapses[nodeIndex] = bpInputNode.synapses[nodeIndex] + currNode.value * bpInputNode.delta * self.learingRate
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
	for i = 1,numOfSynapses do
		synapses[i] = 2 * math.random() - 1
	end
	local newNode = {synapses = synapses, values = {}, deltas = {}, synapseAdjustments = {}}
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


math.randomseed(8981)
myNetwork = nn:new({2,6,1},2)
for iteration = 1,300000 do
	myNetwork:propagate({0,0},{0})
	myNetwork:propagate({1,0},{1})
	myNetwork:propagate({0,1},{1})
	myNetwork:propagate({1,1},{0})
end
myNetwork:propagate({0,0},{0})
print("0,0,1 | " .. myNetwork.layers[3].nodes[1].value)
myNetwork:propagate({0,1},{1})
print("0,1,1 | " .. myNetwork.layers[3].nodes[1].value)
myNetwork:propagate({1,0},{1})
print("1,0,1 | " .. myNetwork.layers[3].nodes[1].value)
myNetwork:propagate({1,1},{0})
print("1,1,1 | " .. myNetwork.layers[3].nodes[1].value)
