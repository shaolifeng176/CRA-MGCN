import torch
model_state_dict = torch.load('model_fold0_best.pth')
for key in model_state_dict.keys():
    print(key, model_state_dict[key])