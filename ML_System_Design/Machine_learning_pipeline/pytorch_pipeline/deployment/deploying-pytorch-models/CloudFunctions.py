import os
import numpy as np
import requests

import torch
from torch import nn
from torch.nn import functional as F

def wine_classifier(request):

    MODEL_URL = 'https://storage.googleapis.com/pytorch-models/classifier_state_dict_'
    r = requests.get(MODEL_URL)
    

    
    file = open("/tmp/model.pth", "wb")
    file.write(r.content)
    file.close()
    
    # State dict requires model object
    
    input_size = 13
    output_size = 3
    hidden_size = 100

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)

        def forward(self, X):
            X = torch.sigmoid((self.fc1(X)))
            X = torch.sigmoid(self.fc2(X))
            X = self.fc3(X)

            return F.log_softmax(X, dim=-1)

    
    model = Net()
    model.load_state_dict(torch.load('/tmp/model.pth'))
    
    ## Extracting parameter and returning prediction
    if request.method == 'GET':
        return " Welcome to wine classifier"
        
    if request.method == 'POST':

        data = request.get_json()
    
        x_data = data['x']

        sample = np.array(x_data)

        sample_tensor = torch.from_numpy(sample).float()


        out = model(sample_tensor)

        _, predicted = torch.max(out.data, -1)

        pred_class = "The wine belongs to class - " + str(predicted.item())
    
    return pred_class

requests==2.18.4
https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl ## Torch installation for Linux. Pip install torch installs some torch which causes problems on the cloud function vm. https://stackoverflow.com/questions/55449313/google-cloud-function-python-3-7-requirements-txt-makes-deploy-fail
numpy==1.16.1

curl -XPOST https://us-central1-loony-csd-project.cloudfunctions.net/function-1 -H 'Content-Type: application/json' -d @test_data0.json
