require('mininet')
nums = {}
count = 0
isDivisable = {}
divisor = 4
for bit1 = 0,1 do
	for bit2 = 0,1 do
		for bit3 = 0,1 do
			for bit4 = 0,1 do
				for bit5 = 0,1 do
					for bit6 = 0,1 do
						for bit7 = 0,1 do
							for bit8 = 0,1 do
								nums[count] = {bit1,bit2,bit3,bit4,bit5,bit6,bit7,bit8}
								if count % divisor == 0 then
									isDivisable[count] = {1}
								else
									isDivisable[count] = {0}
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
--print(time)
math.randomseed(1)

batchSize = 8
local net = mininet:new({8,14,1},1)
local iters = 10000
for iteration = 1,iters do
	-- batch by 32
	i = 0
	j = i + batchSize - 1
	repeat
		net:backPropagate({unpack(nums,i,j)},{unpack(isDivisable,i,j)})
		i = j + 1
		j = i + batchSize - 1
	until j > #nums
	
	if iteration % 1000 == 0 then
		print(iteration / iters * 100 .. "%" .. " Time left: " .. os.clock() * (iters / iteration) - os.clock() .. " seconds")
	end
end
zeroesCorrect = 0
onesCorrect = 0
totalOnes = 0
for count = 0,#nums do
	local guess = net:predictOutputs({nums[count]})[1][1]
	local answer = 0
	if count % divisor == 0 then
		answer = 1
		totalOnes = totalOnes + 1
	end
	print(answer .. " | " .. guess)
	if answer == 1 and guess > 0.5 then onesCorrect = onesCorrect + 1 end
	if answer == 0 and guess < 0.5 then zeroesCorrect = zeroesCorrect + 1 end
end
print("Accuracy for guessing ones: " .. onesCorrect / totalOnes * 100 .. "%")
print("Accuracy for guessing zeroes: " .. zeroesCorrect / (#nums + 1 - totalOnes) * 100 .. "%")
