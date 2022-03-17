import os
# import nibabel as nib
from glob import glob
from torch.utils.data import Dataset
# import h5py
import sirf.Gadgetron as pMR


class ImageDataset(Dataset):
    """ Dataset class for torch data loader """
    def __init__(self, data_dir, transform=None):
        
        self.data_dir = data_dir
        self.files = glob(f"{self.data_dir}y_*.h5")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        # img_path = os.path.join(self.data_dir, f"img_{idx}.nii")
        # img = nib.load(img_path)
        # img = h5py.File(img_path, "r")
        
        data_path = os.path.join(self.data_dir, f"y_{idx}.h5")
        acq_data = pMR.AcquisitionData(data_path)
        
        if self.transform:
            img = self.transform(img)
            
        return acq_data