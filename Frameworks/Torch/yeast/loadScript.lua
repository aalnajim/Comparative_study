
require 'nn'


local trainer = require(".\model")
function string:splitAtCommas()
  local sep, values = ",", {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(c) values[#values+1] = c end)
  return values
end

function loadData(dataFile)
  local dataset = {}
  i = 0
  for line in io.lines(dataFile) do
    local values = line:splitAtCommas()
    local y = torch.Tensor(10)
      for j=1, 10 do
	    y[j] = values[j+7] -- the target class is the last number in the line
      end
    local x = torch.Tensor(7) -- the input data is all the other numbers
      for j=1, 7 do
	    x[j] = values[j] -- the target class is the last number in the line
	  end
	dataset[i] = {x, y}
    i = i + 1
  end
  function dataset:size() return (i - 1) end -- the requirement mentioned
  return dataset
end
function normalize(value)
  local max = 1 
  for i = 2,  10 do 
      if value[i]> value[max] then 
	    max = i
	  end
  end
  for i = 1, 10 do 
    if i == max then
      value [i] 	= 1
  	else 
	  value [i] = 0
	end
  end
  return value
end
  
function compare(x,y) 
   for i =1, 10 do
      if x[i]==y[i] then
	    x[i]=x[i]
	  else
	     return 0 
	   end
   end
   return 1 
end  
function trainModel1()
   local Training_dataset = loadData('yeast.data')
   print('training')
   local trained_Model = trainModel(Training_dataset)
   print('training_completed')
   local test_dataset =  loadData('ecoli.data')
   
   trained_Model:evaluate()
   print('eval')
   local correction = 0   
   for datapoint = 1, #test_dataset, 1 do
	--print(datapoint)
     prediction = trained_Model:forward(test_dataset[datapoint][1])
	 correction = correction + compare (normalize(prediction),test_dataset[datapoint][2])
   end
   print(correction)
   for datapoint = 1, #Training_dataset, 1 do
	--print(datapoint)
     prediction = trained_Model:forward(Training_dataset[datapoint][1])
	 correction = correction + compare (normalize(prediction),Training_dataset[datapoint][2])
   end
   print(correction)
   print('Accuracy is ')
   print(correction/(#test_dataset+#Training_dataset)*100)
end
   