import torch
import torch.nn as nn

num_steps = 1000
batch_size = 1
num_inputs = 2  # traffic densities: (NS and EW)
num_outputs = 2  # green light duration: (NS and EW)
traffic_data = torch.rand(num_steps, batch_size, num_inputs)
targets = 1 - traffic_data.view(num_steps, batch_size, -1)  # Inverse of traffic density as target
class SimpleNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        self.relu = nn.ReLU() # nonlinearity for better pattern learning
        self.layer_norm1 = nn.LayerNorm(64) # normalize layer inputs
        self.dropout = nn.Dropout(p=0.5) # prevent overfitting
        self.fc2 = nn.Linear(64, 32)
        self.layer_norm2 = nn.LayerNorm(32)
        self.fc3 = nn.Linear(32, num_outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.layer_norm1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
    '''training to minimize loss function by updating tensor parameters 
    (NC generalizes to real-time adjustments and models how NC devices will be able to update information)'''
    def train(self, optimizer, loss_fn, traffic_data):
        for step in range(num_steps):
            optimizer.zero_grad()
            input_tensor = traffic_data[step].view(batch_size, -1)  
            output = self(input_tensor) # call forward method
            
            # target is the inverse of traffic density (more traffic -> longer green light)
            target = 1 - traffic_data[step].view(batch_size, -1)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                print(f"Step {step}, Loss: {loss.item()}")

        return output

def test(): 
    model = SimpleNN(num_inputs, num_outputs)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    output = model.train(optimizer=optimizer, loss_fn=loss_fn, traffic_data=traffic_data)


    print("Output data shape:", output.shape)
    print("Output data:", output.detach().numpy()) # output is relative green light durations for NS vs EW intersections

test()