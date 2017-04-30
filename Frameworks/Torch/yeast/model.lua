
require 'nn'

function trainModel(dataset) 
  -- Define model architecture ---------------------------------------------------
  input_layer=7
  output_layer=10
  hidden_layer=4
  first_layer = nn.Linear(input_layer,hidden_layer)
  out_layer = nn.Linear(hidden_layer,output_layer)
  model = nn.Sequential()
  model:add(first_layer)
  model:add(nn.Sigmoid())
  model:add(out_layer)
  model:add(nn.Sigmoid())
-- Trainer definition ----------------------------------------------------------
  criterion = nn.MSECriterion()
  trainer = nn.StochasticGradient(model, criterion)
  trainer.learningRate = 0.1
  trainer.maxIteration = 200
-- Training --------------------------------------------------------------------
  trainer:train(dataset)
  torch.save('YeastModel',model)
  return model
 end
