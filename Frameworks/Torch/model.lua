
require 'nn'



dataset = {}
function dataset:size() return 50 end
x = torch.linspace(-1,1,dataset:size())
y = x:clone():pow(2)
for i = 1, dataset:size() do
   dataset[i] = {x:reshape(x:size(1),1)[i], y:reshape(y:size(1),1)[i]}
end

-- Define model architecture ---------------------------------------------------
input_layer=7
output_layer=10
hidden_layer=3
first_layer = nn.Linear(input_layer,hidden_layer)
out_layer = nn.Linear(hidden_layer,output_layer)
model = nn.Sequential()
model:add(first_layer)
model:add(sig)
model:add(out_layer)

-- Trainer definition ----------------------------------------------------------
criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.01

-- Training --------------------------------------------------------------------
trainer:train(dataset)
