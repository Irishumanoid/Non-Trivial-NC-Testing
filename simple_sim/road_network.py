import numpy as np
from intersection_predict import SimpleNN
import torch
import torch.nn as nn
from itertools import product
from random import sample

#tensors represent time at NS to be green, then switches to SW (frac NS/total*1 minute, frac EW/total*1 minute)
#simulation: grid with vert and hor lines at each intersection, timer in corner
# TODO: integrate bool to determine whether cycle starts with NS or EW green light
num_steps = 1000
batch_size = 1
num_inputs = 2  # traffic densities: (NS and EW)
num_outputs = 2  # green light duration: (NS and EW)

'''
- a bunch of NNs from intersection_predict.py placed randomly in a grid to simulate a city (assuming only 2 road intersections)
- traffic light durations adaptively tuned according to nearby light durations
'''
class RoadNetwork():
    def __init__(self, grid_dimensions: tuple, num_intersections):
        self.x_max = grid_dimensions[0]
        self.y_max = grid_dimensions[1]
        self.num_intersections = num_intersections
        self.product_list = list(product(range(self.x_max), range(self.y_max)))
        self.intersections = sample(self.product_list, k=self.num_intersections)
        self.weights = {} 
        self.loc = np.zeros((self.x_max, self.y_max), dtype=object)
        self.loss_fn = nn.MSELoss()

    # after weights are adjusted for 1 interval, regenerate traffic (generates distinct tuples for intersection coordinates)
    def assign_intersections(self):
        for i in self.intersections:
            self.loc[i[0]][i[1]] = SimpleNN(num_inputs, num_outputs)
    
    def train(self):
        self.weights = {}  
        for i in self.intersections:
            model = self.loc[i[0]][i[1]]
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            cur_traffic_data = torch.rand(num_steps, batch_size, num_inputs)
            out = model.train(optimizer, self.loss_fn, cur_traffic_data)
            self.weights[(i[0], i[1])] = out
        
        return self.weights
    
    # run first interval with init weights, then change tensors weights based on the weights of tensors in the same row or column
    def adjust_row_weights(self):
        rows = set([tup[0] for tup in self.intersections])
        cols = set([tup[1] for tup in self.intersections])

        # for a given row, if one row has a col_val >> row_val, increase other col vals in same row to reduce congestion
        for row in rows:
            entries = [tup for tup in self.intersections if tup[0] == row]
            if len(entries) > 1:
                row_weights = [self.weights[(row, col)] for col in [tup[1] for tup in entries]]
                weights_list = [tensor.detach().tolist()[0] for tensor in row_weights]
                print(f"Row {row} weights_list: {weights_list}")

                max_diff = max([entry[1] - entry[0] for entry in weights_list])
                print(f"Max difference for row {row}: {max_diff}")

                if max_diff > 0.1:
                    for i, old_w in enumerate(weights_list):
                        if old_w[1] - old_w[0] != max_diff:
                            updated_w = [old_w[0], old_w[1] + max_diff / 2]
                            print(f"Updating weights at {(row, entries[i][1])} with new values {updated_w}")
                            self.weights[(row, entries[i][1])] = torch.tensor([updated_w], dtype=torch.float32, requires_grad=True)


        # for a given col, if one col has a row_val >> col_val, increase other row vals in the same column
        for col in cols:
            entries = [tup for tup in self.intersections if tup[1] == col]
            if len(entries) > 1:
                col_weights = [self.weights[(row, col)] for row in [tup[0] for tup in entries]]
                weights_list = [tensor.detach().tolist()[0] for tensor in col_weights]
                print(f"Col {col} weights_list: {weights_list}")

                max_diff = max([entry[0] - entry[1] for entry in weights_list])
                print(f"Max difference for col {col}: {max_diff}")

                if max_diff > 0.1:
                    for i, old_w in enumerate(weights_list):
                        if old_w[0] - old_w[1] != max_diff:
                            updated_w = [old_w[0] + max_diff / 2, old_w[1]]
                            print(f"Updating weights at {(entries[i][0], col)} with new values {updated_w}")
                            self.weights[(entries[i][0], col)] = torch.tensor([updated_w], dtype=torch.float32, requires_grad=True)


    # outputs array of arrays of tensors at each intersection for each timestamp
    def simulation_contents(self, time):
        splits = np.zeros(time, dtype=object)
        for i in range(len(splits)):
            self.assign_intersections()
            splits[i] = self.train()
            self.adjust_row_weights()

        return splits


network = RoadNetwork((5, 4), 7)

contents = network.simulation_contents(5)
print('*'*50)
for c in contents:
    print(c)