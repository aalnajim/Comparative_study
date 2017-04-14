
local trainer = require("scripts.model")
function string:splitAtCommas()
  local sep, values = ",", {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(c) values[#values+1] = c end)
  return values
end

function loadData(dataFile)
  local dataset = {}
  for line in io.lines(dataFile) do
    local values = line:splitAtCommas()
    local y = torch.Tensor(10)
      for j=1, 10 do
	    y[j] = values[j+7] -- the target class is the last number in the line
    
    local x = torch.Tensor(7) -- the input data is all the other numbers
      for j=1, 7 do
	    x[j] = values[j] -- the target class is the last number in the line
	dataset[i] = {x, y}
    i = i + 1
  end
  function dataset:size() return (i - 1) end -- the requirement mentioned
  return dataset
end

function trainModel()
   local Training_dataset = loadData('yeast.data')
   local trained_Model = trainer.trainModel(Training_dataset)
   local test_dataset =  loadData('ecoli.data')
   