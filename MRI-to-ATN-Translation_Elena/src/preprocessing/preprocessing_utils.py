import importlib, subprocess, sys

# Check libraries
with open("requirements.txt") as f:
    for line in f:
        pkg = line.strip().split("==")[0]
        try:
            importlib.import_module(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", line.strip()])

import os
import tqdm
import ants
import glob
import warnings
import numpy as np
import pandas as pd
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from antspynet.utilities import brain_extraction
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

mni_template = ants.image_read("../../templates/TPM_GM.nii")    

def preprocess_mri(img):
    """
    Skull-stripping, including storing a brainmask,
    and rigid (atrophy-preserving) coregistration to MNI space.
    Final images are stored as 121x145x121 Nifti files.
    
    Parameters:
        image_list (list): List containing image paths.
        plotting (boolean): Whether transformed MRI scans should be plotted.
        
    Returns:
        dict: A dictionary comprising the mask, the transformed MRI
        and transformation parameters
    """
    
    # Load MRI scan
    path_tmp = os.path.dirname(img)
    filename_tmp = os.path.basename(img).split(".")[0]+".nii"
    img_ants = ants.image_read(img)
    
    # Rigid-only registration to TPM (to preserve atrophy patterns)
    rigid_params = ants.registration(
        fixed=mni_template,
        moving=img_ants,
        type_of_transform='Rigid',
        random_seed=0,
        outprefix="./rmdir/"
    )
    rigid_mri = rigid_params['warpedmovout']
    
    # Skull stripping using tensorflow-based function
    brain_extracted = brain_extraction(rigid_mri, modality='t1')
    rigid_mask = ants.get_mask(brain_extracted, low_thresh=0.5)
    rigid_mask = ants.morphology(rigid_mask, "erode", radius=5)
    rigid_mask = ants.morphology(rigid_mask, "open", radius=5)
    rigid_mask = ants.morphology(rigid_mask, "dilate", radius=5)
    rigid_mask = ants.morphology(rigid_mask, "close", radius=5)
    
    rigid_mask = ants.threshold_image(rigid_mask, 0.5, 1e9)
    
    assert rigid_mri.shape == rigid_mask.shape, \
    f"Mismatch: MRI {rigid_mri.shape}, Mask {rigid_mask.shape}"
    
    # Apply mask to transformed MRI
    rigid_masked = rigid_mri * rigid_mask
    
    # Remove values below 0 from MRI
    rigid_masked = rigid_masked * (rigid_masked > 0)

    # Store results
    if not os.path.exists(path_tmp+"/preprocessed/"):
        os.makedirs(path_tmp+"/preprocessed/")
    ants.image_write(rigid_mask, f'{path_tmp}/preprocessed/m_{filename_tmp}')
    ants.image_write(rigid_masked, f'{path_tmp}/preprocessed/r_{filename_tmp}')

    params = {'mask': rigid_mask, 'mri': rigid_masked, 'params': rigid_params,
             'mri_unmasked': rigid_mri}

    return params
    
def preprocess_pet(pet, mri_unmasked, smoothing, mask):
    """
    Exact co-registration of PET to MRI,
    smoothing of PET scans
    and skull-stripping based on MRI brainmask.
    Final images are stored as Nifti files in shape of MRI.
    
    Parameters:
        pet (ANTsImage): PET scan in native space
        mri_unmasked (ANTsImage): MRI scan that was rigidly coregistered to MNI space
        smoothing (int): Intensity of smoothing. Common values are 2, 4, 8 and 16.
        mask (ANTsImage): Binary brain mask derived from MRI
        
    Returns:
        ANTsImage: Coregistered, smoothed and skull-stripped PET scan
    """
    # Get image information for storage of results
    path_tmp = os.path.dirname(pet)
    filename_tmp = os.path.basename(pet).split(".")[0]+".nii"
    
    # Load PET scan
    img_ants = ants.image_read(pet)
    
    # Exact registration of PET to MRI
    params = ants.registration(
        fixed=mri_unmasked,
        moving=img_ants,
        type_of_transform='SyNAggro',
        random_seed=0,
        outprefix="./rmdir/"
    )
    
    # Smooth PET to remove noise
    smoothed = ants.smooth_image(params['warpedmovout'], sigma=smoothing, FWHM=True)
    
    # Apply mask derived from MRI
    masked_pet = smoothed * mask
    
    # Store results
    if not os.path.exists(path_tmp+"/preprocessed/"):
        os.makedirs(path_tmp+"/preprocessed/")
    ants.image_write(masked_pet, path_tmp+f'/preprocessed/r_s{smoothing}_{filename_tmp}')
    
    return masked_pet

def preprocess_rois(rois, names, mri, mask, folder):
    """
    Exact co-registration of ROIs to MRI,
    ensuring they are binary (if masks) or integer-type (if atlas).
    Final images are stored as Nifti files in shape of MRI.
    
    Parameters:
        rois (list): List containing file location (str) for each ROI
        names (list): List containing name (str) of each ROI
        mask (ANTsImage): Binary brain mask derived from MRI
        folder (str): Location in which to store the transformed ROIs
        
    Returns:
        dict: Dictionary containing the transformed ROIs
    """
    results = {n: [] for n in names}
    
    # Loop through ROIs
    for i in range(len(rois)):
        r = rois[i]
        
        # Load image
        img_ants = ants.image_read(r)
        
        # Exact matching of ROIs to MRI
        params = ants.registration(
                fixed=mri,
                moving=mni_template,
                type_of_transform="SyNAggro",
                random_seed=0,
                outprefix="./rmdir/")
        # Apply transforms, ensure value similarity to org through nearestNeighbor
        transformed_roi = ants.apply_transforms(
                fixed=mri,
                moving=img_ants,
                transformlist=params['fwdtransforms'],
                interpolator='nearestNeighbor')
        
        # Transform to strictly binary if not an atlas
        if "atlas" not in names[i]:
            transformed_roi = ants.threshold_image(transformed_roi, 0.5, 1e9)
        # Transform to integer values if atlas
        else:
            transformed_roi = transformed_roi.astype('uint32')

        # Apply mask to remove voxels outside of the brain
        masked_roi = transformed_roi * mask
        results[names[i]] = masked_roi

        # Store results
        if not os.path.exists(folder+"/preprocessed/"):
            os.makedirs(folder+"/preprocessed/")
        ants.image_write(masked_roi, folder+f'/preprocessed/{names[i]}.nii')

    return results

def compute_suvr(pet, roi, folder, img_id, smoothing=8):
    """
    Compute SUVr maps from PET.
    Final images are stored as Nifti files in the shape of PET input.
    
    Parameters:
        pet (ATNsImage): PET scan for which to compute SUVr map
        roi (ANTsImage): Reference region (must be in same space and dimension as pet)
        folder (str): Location in which to store SUVr map
        img_id (str): ID under which to store SUVr map
        
    Returns:
        ANTsImage: SUVr map in shape of PET input
        
    """
    assert pet.shape == roi.shape, f"Shape mismatch - PET {pet.shape}, reference {roi.shape}"
    # Divide by mean intensity in reference region
    roi_pet_mean = pet[roi>0].mean()    
    pet = pet / roi_pet_mean

    # Store results
    ants.image_write(pet, folder+f'/preprocessed/SUVr_{img_id}.nii')

    return pet