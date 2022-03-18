import os
import nibabel as nib
import sirf.Gadgetron as pMR
from glob import glob
from torch.utils.data import Dataset


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
        
        self.transform = transform

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        
        label_path = os.path.join(self.label_dir, f"img_{idx}.nii")
        label_data = nib.load(label_path)
        
        input_path = os.path.join(self.input_dir, f"y_{idx}.h5")
        input_data = pMR.AcquisitionData(input_path)
        
        # TODO: Create an acquisition model from acquisition data object
        
        # if self.transform:
        #    img = self.transform(img)
            
        return input_data, label_data