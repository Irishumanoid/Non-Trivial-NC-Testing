import argparse
import os
import sys
import torch
import lava.lib.dl.slayer as slayer
from torch.utils.data import DataLoader

from net_utils.utils import TrafficDataset
from net_utils.snns import SlayerDenseSNN, LavaDenseSNN

from PIL import Image
import numpy as np

def get_all_files(path, keyword):
  paths = []
  for root, dirs, files in os.walk(path):
    for name in files:
      if str(name).__contains__(keyword):
            paths.append(os.path.abspath(os.path.join(root, name)))
  return paths

class TrainEvalSNN():
  def __init__(self, device, epochs, n_tsteps):
    self.model = SlayerDenseSNN().to(device)
    self.device = device
    self.epochs = epochs
    self.n_ts = n_tsteps


  def train_eval_snn(self):
    loss = slayer.loss.SpikeRate(
        # `true_rate` and `false_rate` should be between [0, 1].
        true_rate=0.9, # Keep `true_rate` high for quicker learning.
        false_rate=0.01, # Keep `false_rate` low for quicker learning.
        reduction="sum").to(self.device)
    stats = slayer.utils.LearningStats()
    optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    assistant = slayer.utils.Assistant(
        self.model, loss, optimizer, stats,
        classifier=slayer.classifier.Rate.predict)

    for epoch in range(1, self.epochs+1):

      self.model.train()
      image_paths = get_all_files(r'/Users/irislitiu/Downloads/traffic_dataset_labeled/train/images', '.jpg')
      label_paths = get_all_files(r'/Users/irislitiu/Downloads/traffic_dataset_labeled/train/labels', '.txt')
      train_data = TrafficDataset(image_paths=image_paths, vehicle_label_paths=label_paths, n_tsteps=self.n_ts)
      train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
      for inp, lbl in train_loader:
        inp, lbl = inp.to(self.device), lbl.to(self.device)
        output = assistant.train(inp, lbl)

      self.model.eval()
      image_paths = get_all_files(r'/Users/irislitiu/Downloads/traffic_dataset_labeled/test/images', '.jpg')
      label_paths = get_all_files(r'/Users/irislitiu/Downloads/traffic_dataset_labeled/test/labels', '.txt')
      test_data = TrafficDataset(image_paths=image_paths, vehicle_label_paths=label_paths, n_tsteps=self.n_ts)
      test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
      for inp, lbl in test_loader:
        inp, lbl = inp.to(self.device), lbl.to(self.device)
        output = assistant.test(inp, lbl)

      print("Epoch: {0}, Stats: {1}".format(epoch, stats))
      if stats.testing.best_accuracy:
        torch.save(self.model.state_dict(), "./trained_traffic_network.pt")
      stats.update()

    self.model.load_state_dict(torch.load("./trained_traffic_network.pt"))
    self.model.export_hdf5("./trained_traffic_network.net")
  
  """
  Preprocess an image for inference based on hte number of time steps for spike generation, returning the spikes generated.
  """
  def preprocess_image(self, image_path, n_tsteps):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img = np.array(img) / 255.0  # Normalize
    v = np.zeros_like(img)
    spikes = np.zeros((img.shape[0] * img.shape[1], n_tsteps), dtype=np.float32)
    for t in range(n_tsteps):
        J = img  # Assuming gain=1 and bias=0
        v = v + J
        mask = v > 1.0
        spikes[:, t] = mask.flatten().astype(np.float32)
        v[mask] = 0
    return torch.tensor(spikes).to(self.device)

  """
  Predict the class of a certain image (0-9)
  """
  def predict_image(self, image_path):
      spikes = self.preprocess_image(image_path, self.n_ts)
      spikes = spikes.unsqueeze(0)  # Add batch dimension
      self.model.eval()
      with torch.no_grad():
          output = self.model(spikes)
      predicted_class = output.sum(dim=-1).argmax().item()
      return predicted_class

  def predict_images(self, image_paths):
        for image_path in image_paths:
          self.predict_image(self, image_path)

if __name__=="__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"
  parser = argparse.ArgumentParser()
  parser.add_argument("--epochs",type=int, default=25, required=False)
  parser.add_argument("--n_tsteps", type=int, default=32, required=False)
  parser.add_argument("--backend", type=str, default="GPU", required=False)
  parser.add_argument("--num_test_imgs", type=int, default=25, required=False)
  parser.add_argument("--image_path", type=str, default="", required=False)

  args = parser.parse_args()

  tes = TrainEvalSNN(device=device, epochs=args.epochs, n_tsteps=args.n_tsteps)
  if args.image_path:
    print("*" * 80)
    print(f"Making predictions for image: {args.image_path}")
    print("*" * 80)

    if os.path.isfile('./trained_traffic_network.pt'):  
      tes.model.load_state_dict(torch.load("./trained_traffic_network.pt"))
    else:
      print("Trained model weights not found. Train the model before running any predictions.")
      sys.exit(1)
    
    predicted_class = tes.predict_image(args.image_path)
    print(f"Predicted class for the image {args.image_path}: {predicted_class}")
    sys.exit(0)

  if args.backend == "GPU":
    print("*"*80)
    print("Training and Evaluating SlayerDenseSNN on GPU ... AND,\n"
          "Evaluating LavaDenseSNN on Loihi-2 Simulation Hardware on CPU.")
    print("*"*80)
    tes.train_eval_snn()
    #TODO might need to adjust these
    lava_snn = LavaDenseSNN(
        "./trained_traffic_network.net",
        img_shape=784,
        n_tsteps=args.n_tsteps,
        st_img_id=0, # Start evaluating from the 1st test image.
        num_test_imgs=args.num_test_imgs,
        )
    lava_snn.infer_on_loihi(backend="L2Sim")

  elif args.backend == "L2Sim" or args.backend == "L2Hw":
    try:
      assert os.path.isfile("./trained_traffic_network.net")
    except:
      print("*"*80)
      sys.exit(
          "First train SlayerDenseSNN on GPU to obtain trained weights. Exit.. "
          "\n"+"*"*80)

    if args.backend == "L2Sim":
      print("*"*80)
      print("Only evaluating the LavaDenseSNN on Loihi-2 Simulation Hardware.")
      print("*"*80)
    elif args.backend == "L2Hw":
      print("*"*80)
      print("Only evaluating the LavaDenseSNN on Loihi-2 Physical Hardware.")
      print("*"*80)
    lava_snn = LavaDenseSNN(
        "./trained_traffic_network.net",
        img_shape=784,
        n_tsteps=args.n_tsteps,
        st_img_id=0, # Start evaluating from the 1st test image.
        num_test_imgs=args.num_test_imgs,
        )
    lava_snn.infer_on_loihi(args.backend)



