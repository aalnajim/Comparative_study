
require 'nn'

function trainModel(dataset) 
  -- Define model architecture ---------------------------------------------------
  input_layer=6
  output_layer=1
  hidden_layer1=4
  hidden_layer2=4
  first_layer = nn.Linear(input_layer,hidden_layer1)
  second_layer = nn.Linear(hidden_layer1,hidden_layer2)
  out_layer = nn.Linear(hidden_layer2,output_layer)
  model = nn.Sequential()
  model:add(first_layer)
  model:add(nn.Sigmoid())
  model:add(second_layer)
  model:add(nn.Sigmoid())
  model:add(out_layer)
  model:add(nn.Sigmoid())
-- Trainer definition ----------------------------------------------------------
  criterion = nn.MSECriterion()
  trainer = nn.StochasticGradient(model, criterion)
  trainer.learningRate = 0.01
  trainer.maxIterations = 200
-- Training --------------------------------------------------------------------
  trainer:train(dataset)
  torch.save('CarClassifier',model)
  return model
 end
