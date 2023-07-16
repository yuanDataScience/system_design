import os
import numpy as np
from flask import Flask, request, jsonify 
import requests

import torch
from torch import nn
from torch.nn import functional as F


MODEL_URL = 'https://storage.googleapis.com/pytorch-models/classifier_state_dict_'

PORT = 8080

r = requests.get(MODEL_URL)

file = open("models/model_state_dict.pth", "wb")
file.write(r.content)
file.close()


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

model.load_state_dict(torch.load('models/model_state_dict.pth'))


app = Flask(__name__)

@app.route("/")
def hello():
    return "Binary classification example\n"

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    data = request.get_json()
    
    x_data = data['x']
    
    sample = np.array(x_data)

    sample_tensor = torch.from_numpy(sample).float()


    out = model(sample_tensor)

    _, predicted = torch.max(out.data, -1)

    pred_class = "The wine belongs to class - " + str(predicted.item())

    return jsonify(pred_class)


if __name__ == '__main__':

    app.run(debug=True, port=PORT, use_reloader=False)
    
#curl -XPOST http://127.0.0.1:8080/predict -H 'Content-Type: application/json' -d @test_data1.json