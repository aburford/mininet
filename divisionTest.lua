require('nn')
nums = {}
count = 0
isDivisable = {}
for bit1 = 0,1 do
	for bit2 = 0,1 do
		for bit3 = 0,1 do
			for bit4 = 0,1 do
				for bit5 = 0,1 do
					for bit6 = 0,1 do
						for bit7 = 0,1 do
							for bit8 = 0,1 do
								nums[count] = {bit7,bit8}
								if count % 4 == 0 then
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
print(time)
math.randomseed(3)

for batchSize = 1, 100 do
	net = nn:new({2,3,1},1)
	iters = 10000
	for iteration = 1,iters do
		-- batch by 32
		i = 1
		j = i + batchSize
		repeat
			net:backPropagate({unpack(nums,i,j)},{unpack(isDivisable,i,j)})
			j = i + batchSize
			i = j + 1
		until j > #nums
		
		--[[if iteration % 10 == 0 then
			print(iteration / iters * 100 .. "%" .. " Time left: " .. os.clock() * (iters / iteration) - os.clock() .. " seconds")
		end]]
	end
	net:backPropagate({nums[120]},{{1}})
	print(batchSize .. "," .. net.layers[3].nodes[1].values[1])
end