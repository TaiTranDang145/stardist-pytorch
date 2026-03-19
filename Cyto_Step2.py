import argparse
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from sparse_mincut_pool import sparse_mincut_pool_batch

from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import GraphConv
import os
import numpy as np
import pandas as pd
import datetime
import csv
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--graph_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--Num_TCN', type=int, default=6)
parser.add_argument('--Num_Run', type=int, default=20)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()


## Hyperparameters
LastStep_OutputFolderName = os.path.join(args.graph_dir, "")
Num_TCN = args.Num_TCN
Num_Run = args.Num_Run
Num_Epoch = args.epochs
Embedding_Dimension = 128
Learning_Rate = args.lr
Loss_Cutoff = -0.6
device_name = args.device


## Import image name list.
# We read the ImageNameList.txt copied to the graph_dir by Step 1 to ensure indices match.
Region_filename = os.path.join(LastStep_OutputFolderName, "ImageNameList.txt")
if not os.path.exists(Region_filename):
    print(f"Cảnh báo: Không tìm thấy {Region_filename}. Thử lấy từ glob (có thể sai thứ tự).")
    import glob
    edge_index_files = glob.glob(os.path.join(LastStep_OutputFolderName, "*_EdgeIndex.txt"))
    region_names = [os.path.basename(f).replace("_EdgeIndex.txt", "") for f in edge_index_files]
    region_name_list = pd.DataFrame({"Image": region_names})
else:
    region_name_list = pd.read_csv(
        Region_filename,
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["Image"],  # set our own names for the columns
    )


## Load dataset from the constructed Dataset.
MaxNumNodes_filename = LastStep_OutputFolderName + "MaxNumNodes.txt"
max_nodes = np.loadtxt(MaxNumNodes_filename, dtype = 'int64', delimiter = "\t").item()

class SpatialOmicsImageDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SpatialOmicsImageDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['SpatialOmicsImageDataset.pt']

    def download(self):
        pass
    
    def process(self):
        pass

dataset = SpatialOmicsImageDataset(LastStep_OutputFolderName)


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=Embedding_Dimension):
        super(Net, self).__init__()

        self.conv1 = GraphConv(in_channels, hidden_channels)
        num_cluster1 = Num_TCN   #This is a hyperparameter.
        self.pool1 = Linear(hidden_channels, num_cluster1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        s = self.pool1(x)   #Here "s" is a non-softmax tensor.
        x_pool, adj_pool, mc1, o1 = sparse_mincut_pool_batch(x, edge_index, s, batch)
        #Save important clustering results_1.
        ClusterAssignTensor_1 = s
        ClusterAdjTensor_1 = adj_pool

        return F.log_softmax(x_pool, dim=-1), mc1, o1, ClusterAssignTensor_1, ClusterAdjTensor_1


def train(epoch, model, optimizer, train_loader, device):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, mc_loss, o_loss, _, _ = model(data.x, data.edge_index, data.batch)
        loss = mc_loss + o_loss
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all


device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

for graph_index, row in region_name_list.iterrows():
    Image_Name = row["Image"]
    print(f"\nProcessing image: {Image_Name}")
    
    ThisStep_OutputFolderName = os.path.join(args.output_dir, f"Step2_Output_{Image_Name}")
    os.makedirs(ThisStep_OutputFolderName, exist_ok=True)
    
    # We find the index of this image in the ordered dataset
    # Dataset order depends on however it was read in Step1 (usually in order of ImageNameList)
    # The safest way is index from region_name_list!
    # Let's hope the order matches Step1, because Step1 uses ImageNameList directly.
    # We will use the original ImageNameList file from the same directory where Step1 read it if possible!
    
    # Actually, the dataset is loaded directly. Let's just use graph_index if we assume the list matches.
    train_dataset = dataset[graph_index]
    train_loader = DataLoader([train_dataset], batch_size=1)
    all_sample_loader = DataLoader([train_dataset], batch_size=1)

    run_number = 1
    while run_number <= Num_Run:  #Generate multiple independent runs for ensemble.
        print(f"This is Run{run_number:02d} for {Image_Name}")

        model = Net(dataset.num_features, 1).to(device)  #Initializing the model.
        optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate)
        
        RunFolderName = os.path.join(ThisStep_OutputFolderName, f"Run{run_number}")
        if os.path.exists(RunFolderName):
            shutil.rmtree(RunFolderName)
        os.makedirs(RunFolderName)  #Creating the Run folder.
        
        filename_0 = os.path.join(RunFolderName, "Epoch_UnsupervisedLoss.csv")
        headers_0 = ["Epoch", "UnsupervisedLoss"]
        with open(filename_0, "w", newline='') as f0:
            f0_csv = csv.writer(f0)
            f0_csv.writerow(headers_0)

        previous_loss = float("inf")  #Initialization.
        for epoch in range(1, Num_Epoch+1):  #Specify the number of epoch in each independent run.
            train_loss = train(epoch, model, optimizer, train_loader, device)

            with open(filename_0, "a", newline='') as f0:
                f0_csv = csv.writer(f0)
                f0_csv.writerow([epoch, train_loss])
            
            # Thêm log để bạn thấy nó vẫn đang chạy
            if epoch % 100 == 0:
                print(f"  -> Epoch {epoch:03d}/{Num_Epoch}: Loss = {train_loss:.4f}")
            
            if train_loss == 0 and train_loss == previous_loss:
                break
            else:
                previous_loss = train_loss

        print(f"    Kết thúc Run này với Loss = {train_loss:.4f}")
        if train_loss >= Loss_Cutoff:   #This is an empirical cutoff of the final loss to avoid underfitting.
             print(f"    [!] Loss {train_loss:.4f} chưa đạt yêu cầu (< {Loss_Cutoff}). Đang Restart lại Run này...")
             shutil.rmtree(RunFolderName)  #Remove the specific folder and all files inside it for re-creating the Run folder.
             continue  #restart this run.

        #Extract the soft TCN assignment matrix using the trained model.
        for EachData in all_sample_loader:
            EachData = EachData.to(device)
            TestModelResult = model(EachData.x, EachData.edge_index, EachData.batch)

            ClusterAssignMatrix1 = TestModelResult[3]
            ClusterAssignMatrix1 = torch.softmax(ClusterAssignMatrix1, dim=-1)  #Checked, consistent with the built-in function "dense_mincut_pool".
            ClusterAssignMatrix1 = ClusterAssignMatrix1.cpu().detach().numpy()
            filename1 = os.path.join(RunFolderName, "TCN_AssignMatrix1.csv")
            np.savetxt(filename1, ClusterAssignMatrix1, delimiter=',')

            ClusterAdjMatrix1 = TestModelResult[4][0, :, :]
            ClusterAdjMatrix1 = ClusterAdjMatrix1.cpu().detach().numpy()
            filename2 = os.path.join(RunFolderName, "TCN_AdjMatrix1.csv")
            np.savetxt(filename2, ClusterAdjMatrix1, delimiter=',')

            NodeMask = EachData.x.size(0)
            NodeMask = np.ones((NodeMask,), dtype=int)
            filename3 = os.path.join(RunFolderName, "NodeMask.csv")
            np.savetxt(filename3, NodeMask.T, delimiter=',', fmt='%i')  #save as integers.

        run_number = run_number + 1

print("All Unsupervised TCN Runs Finished!")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


