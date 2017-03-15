require('mininet')
nums = {}
count = 0
isDivisable = {}
print("Enter a number to learn its factors: ")
divisor = io.read("*n")
for bit1 = 0,1 do
	for bit2 = 0,1 do
		for bit3 = 0,1 do
			for bit4 = 0,1 do
				for bit5 = 0,1 do
					for bit6 = 0,1 do
						for bit7 = 0,1 do
							for bit8 = 0,1 do
								if count ~= 0 then
									nums[count] = {bit1,bit2,bit3,bit4,bit5,bit6,bit7,bit8}
									if count % divisor == 0 then
										isDivisable[count] = {1}
									else
										isDivisable[count] = {0}
									end
								end
								count = count + 1
							end
						end
					end
				end
			end
		end
	end
end

time = os.time()
print("Using randomseed: " .. time)
math.randomseed(time)

trainSet = {unpack(nums,1,200)}
trainTarget = {unpack(isDivisable,1,200)}
testSet = {unpack(nums,201,255)}
testTarget = {unpack(isDivisable,201,255)}

-- setting batch size as divisor typically works best
batchSize = divisor
local net = mininet:new({8,14,1},1)
local iters = 500
userUpdateInterval = iters / 5
for iteration = 1,iters do
	-- batch by 32
	i = 1
	j = i + batchSize - 1
	repeat
		net:backPropagate({unpack(trainSet,i,j)},{unpack(trainTarget,i,j)})
		i = j + 1
		j = i + batchSize - 1
	until j > #trainSet
	if iteration % userUpdateInterval == 0 then
		local time = os.clock()
		print(iteration / iters * 100 .. "%" .. " Time left: " .. time * (iters / iteration) - time .. " seconds")
	end
end

-- check accuracy on the training data
zeroesCorrect = 0
onesCorrect = 0
totalOnes = 0
for count = 1,#trainSet do
	local guess = net:predictOutputs({trainSet[count]})[1][1]
	local answer = trainTarget[count][1]
	if answer == 1 then
		totalOnes = totalOnes + 1
		if guess > 0.5 then onesCorrect = onesCorrect + 1 end
	elseif guess < 0.5 then zeroesCorrect = zeroesCorrect + 1 end
	--print(answer .. " | " .. guess)
end
print("Training data results:")
print("\tFactors: " .. onesCorrect / totalOnes * 100 .. "%")
print("\tNon-factors: " .. zeroesCorrect / (#trainSet - totalOnes) * 100 .. "%")

-- check accuracy on test data that the network has never seen before
zeroesCorrect = 0
onesCorrect = 0
totalOnes = 0
for count = 1,#testSet do
	local guess = net:predictOutputs({testSet[count]})[1][1]
	local answer = testTarget[count][1]
	if answer == 1 then
		totalOnes = totalOnes + 1
		if guess > 0.5 then onesCorrect = onesCorrect + 1 end
	elseif guess < 0.5 then zeroesCorrect = zeroesCorrect + 1 end
	--print(count + #trainSet .. " | " .. table.concat(testSet[count]) .. " | " .. answer .. " | " .. guess)
end
print("\nTest data results:")
print("\tFactors: " .. onesCorrect / totalOnes * 100 .. "%")
print("\tNon-factors: " .. zeroesCorrect / (#testSet - totalOnes) * 100 .. "%")
net:save("factors-of-" .. divisor .. ".mininet")