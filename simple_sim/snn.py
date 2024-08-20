import snntorch as snn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

num_hidden = 64
num_steps = 20
batch_size = 1
num_inputs = 2  # traffic densities: (NS and EW)
num_outputs = 2  # green light duration: (NS and EW)
beta = 0.95
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# fake traffic data
traffic_data = torch.rand(num_steps, batch_size, num_inputs)
targets = 1 - traffic_data.view(num_steps, batch_size, -1)  


class TrafficNet(nn.Module):
    def __init__(self, num_epochs):
        super(TrafficNet, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)
        self.num_epochs = num_epochs
        self.loss_hist = []

        self.data_loader = DataLoader(TensorDataset(traffic_data, targets), batch_size=batch_size, shuffle=True)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

    def train_network(self):
        for epoch in range(self.num_epochs):
            for data, target in self.data_loader:
                data = data.to(device)
                target = target.to(device)

                self.train() 
                spk_rec, mem_rec = self(data)

                # get loss using the final spiking activity, which is averaged spikes over time
                final_spikes = spk_rec.mean(dim=0)  
                loss_val = self.loss_fn(final_spikes, target)

                self.optimizer.zero_grad()
                loss_val.backward()
                self.optimizer.step()

                self.loss_hist.append(loss_val.item())

                print(f"Epoch {epoch}, Loss: {loss_val.item():.2f}")

        return final_spikes.detach().cpu().numpy()[0]  


def test_traffic_net():
    model = TrafficNet(num_epochs=1).to(device)
    output_data = model.train_network()

    print("Output data:", output_data)  # output is relative green light durations for NS vs EW intersections
    return output_data

test_traffic_net()
