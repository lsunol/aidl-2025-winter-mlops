from typing import List

import torch

from model import RegressionModel


@torch.no_grad()
def predict(input_features: List[float]):
    # load the checkpoint from the correct path
    checkpoint = torch.load('./checkpoints')

    # Instantiate the model and load the state dict
    model = RegressionModel(input_size=len(input_features))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Input features is a list of floats. We have to convert it to tensor of the correct shape
    x = torch.tensor(input_features).unsqueeze(0)

    # Now we have to do the same normalization we did when training:
    x = (x - checkpoint['input_mean']) / checkpoint['input_std']

    # We get the output of the model and we print it
    output = model(x)

    # We have to revert the target normalization that we did when training:
    output = output * checkpoint['target_std'] + checkpoint['target_mean']
    print(f"The predicted price is: ${output.item()*1000:.2f}")
