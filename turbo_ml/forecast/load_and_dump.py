import torch
from turbo_ml.forecast.as_meta_model import Best_Model
model = Best_Model(15, 36)
model.load_state_dict(torch.load('model.pth'))
model = model.to('cpu')
torch.save(model.state_dict(), 'model_cpu.pth')
