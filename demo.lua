require('nn')

time = os.time()
print(time)
math.randomseed(1)
-- 00:1 1489375471
-- oddity: a seed of 1489372202 and 1489374325 requires learningRate < 5
myNetwork = nn:new({3,4,1},2)
for iteration = 1,10000 do
	myNetwork:backPropagate({{0,0,1},{1,0,1},{0,1,1},{1,1,1}},{{1},{0},{0},{0}})
end
myNetwork:predictOutputs({{0,0,1},{1,0,1},{0,1,1},{1,1,1}})
predictions = myNetwork:predictOutputs({{0,0,1},{1,0,1},{0,1,1},{1,1,1}})
print("00 | " .. predictions[1][1])
print("10 | " .. predictions[2][1])
print("01 | " .. predictions[3][1])
print("11 | " .. predictions[4][1])


print "Saving network and reopening"
myNetwork:save("orig")
myNetwork = nil
loadedNet = nn:load("orig")
predictions = loadedNet:predictOutputs({{0,0,1},{1,0,1},{0,1,1},{1,1,1}})
-- predictions[batch][node]
print("00 | " .. predictions[1][1])
print("10 | " .. predictions[2][1])
print("01 | " .. predictions[3][1])
print("11 | " .. predictions[4][1])

print 'another test'
loadedNet:save("new")
loadedNet = nil
anotherNet = nn:load("new")
predictions = anotherNet:predictOutputs({{0,0,1},{1,0,1},{0,1,1},{1,1,1}})
-- predictions[batch][node]
print("00 | " .. predictions[1][1])
print("10 | " .. predictions[2][1])
print("01 | " .. predictions[3][1])
print("11 | " .. predictions[4][1])