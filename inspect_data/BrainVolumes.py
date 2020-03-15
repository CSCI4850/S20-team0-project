import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt

class BrainVolumes:
    
    def __init__(self, v_dir):
        """
        Data Structure: each example folder consists of 4 brain volumes and a segmentation volume
        The files have the same name as their parent directory with their scan type added before the file extension
        ex: parent_dir/parent_dir_flair.nii.gz
        """
        self.volumes_dir      = self._check_dir(v_dir)
        self.dir_base_name    = self._get_dir_base(v_dir)
        self.file_extensions  = [("flair", "_flair.nii.gz"), ("seg", "_seg.nii.gz"), ("t1", "_t1.nii.gz"), ("t1ce", "_t1ce.nii.gz"), ("t2", "_t2.nii.gz")]
        self.volume_data      = []
        self.load_volume_data()
    
    def _check_dir(self, v_dir):
        if os.path.isdir(v_dir):
            if v_dir[-1] == '/':
                v_dir = v_dir[:-1]
            return v_dir
        else:
            return False
        
    def _get_dir_base(self, v_dir):
        if self.volumes_dir:
            if v_dir[-1] == '/':
                v_dir = v_dir[:-1]
            split_dir = v_dir.split("/")
            return split_dir[-1]
        else:
            return False
        
    def load_volume_data(self):
        """
        Combines corresponding slices from each volume in a single dictionary and inserts dictionary into a list
        ex: [ { "flair": <2D numpy array>, 
                "t1"   : <2D numpy array>,
                "t1ce" : <2D numpy array>,
                "t2"   : <2D numpy array>,
                "seg"  : <2D numpy array>,
                "s_id" : 155
                }, ...]
        """            
        if self.volumes_dir:
            # clear existing data
            if len(self.volume_data) > 0:
                self.volume_data.clear()

            # initialize data list
            for i in range(154):
                v_slice = {"flair": None, "t1": None, "t1ce": None, "t2": None, "seg": None, "s_id": i}
                self.volume_data.append(v_slice)
                
            for volume_type in self.file_extensions:
                # load data of each volume type
                volume_dir  = "{}/{}{}".format(self.volumes_dir, self.dir_base_name, volume_type[1]) 
                volume      = nib.load(volume_dir)
                volume_data = volume.get_fdata()
                self._add_data(volume_data, volume_type[0])
            
    def _add_data(self, data, v_type):
        # add each slice of brain volume to volume_data list
        for slice_num in range(154):
            current_slice = data[:,:,slice_num]
            if self.volume_data[slice_num]["s_id"] == slice_num:
                self.volume_data[slice_num][v_type] = current_slice
            
    def show_all_volume_dirs(self):
        for volume_type in self.file_extensions:
            volume_dir  = "{}/{}{}".format(self.volumes_dir, self.dir_base_name, volume_type[1]) 
            print(volume_dir)
        
    def set_new_volume_dir(self, new_dir):
        """
        Use this function to load data from another volume
        """
        self.volumes_dir   = self._check_dir(new_dir)
        self.dir_base_name = self._get_dir_base(new_dir)
        self.load_volume_data()

    def display_slice(self, slice_num, image_type):
        if slice_num < 0:
            return
        if len(self.volume_data) == 0:
            return
        if slice_num < len(self.volume_data):
            plt.imshow(self.volume_data[slice_num][image_type])
            plt.show()
