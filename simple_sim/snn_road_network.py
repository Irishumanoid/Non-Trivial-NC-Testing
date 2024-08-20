from snn import TrafficNet
from road_network import RoadNetwork

#tensors represent time at NS to be green, then switches to SW (frac NS/total*1 minute, frac EW/total*1 minute)
num_steps = 1000
batch_size = 1
num_inputs = 2  # traffic densities: (NS and EW)
num_outputs = 2  # green light duration: (NS and EW)
epochs = 1

class SNNRoadNetwork(RoadNetwork):
    def __init__(self, grid_dimensions: tuple, num_intersections):
        super(SNNRoadNetwork, self).__init__(grid_dimensions, num_intersections)

    def assign_intersections(self):
        for i in self.intersections:
            self.loc[i[0]][i[1]] = TrafficNet(num_epochs=epochs)
    
    def train(self):
        self.weights = {}  
        for i in self.intersections:
            model = self.loc[i[0]][i[1]]
            out = model.train_network()

            self.weights[(i[0], i[1])] = out
        
        return self.weights
    
    def adjust_row_weights(self):
        super().adjust_row_weights()

    def simulation_contents(self, time):
        return super().simulation_contents(time)


network = SNNRoadNetwork((3, 2), 3)

contents = network.simulation_contents(5)
print('*'*50)
for c in contents:
    print(c)