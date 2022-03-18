import os
import nibabel as nib
import sirf.Gadgetron as mr
import torch
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    """ Dataset class for torch data loader """
    def __init__(self, input_data_path, label_data_path, transform=None):
        
        self.input_dir = input_data_path
        self.label_dir = label_data_path
        
        input_files = glob(f"{self.input_dir}y_*.h5")
        label_files = glob(f"{self.label_dir}img_*.nii")
        
        if not len(input_files) == len(label_files):
            raise ValueError(f"Number of inputs and labels don't agree. {len(input_files)} inputs and {len(label_files)} labels.")
        
        self.data_len = len(input_files)
        
        if transform is not None:
            self.transform = transform
        else:     
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        
        label_path = os.path.join(self.label_dir, f"img_{idx}.nii")
        label_data = nib.load(label_path)
        label_tensor = torch.tensor(label_data.get_fdata()).unsqueeze(0)
        
        input_path = os.path.join(self.input_dir, f"y_{idx}.h5")
        input_data = mr.AcquisitionData(input_path)
        
        csm = mr.CoilSensitivityData()
        csm.smoothness = 200 # ?
        csm.calculate(input_data)

        model = mr.AcquisitionModel(input_data, csm)
        model.set_coil_sensitivity_maps(csm)
            
        return input_data, label_tensor, model


def collate_func(batch):
    input, label, model = batch[0]
    return list([input, label, model])