import numpy as np
from torch.utils.data import DataLoader
from loguru import logger

from .dataset import CLIPDataset
from .mvtec import load_mvtec, mvtec_classes
from .mvtec2 import load_mvtec2, mvtec2_classes
from .visa import load_visa, visa_classes
from .mpdd import load_mpdd, mpdd_classes
from .dagm import load_dagm, dagm_classes
from .btad import load_btad, btad_classes
from .dtd import load_dtd, dtd_classes
from .medical import load_br35h, load_brainmri, load_brain_tumor_mri, br35h_classes, brainmri_classes, brain_tumor_mri_classes
from .isic import load_isic, isic_classes
from .colonoscopy import load_clinicdb, load_colondb, clinicdb_classes, colondb_classes


mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]

load_function_dict = {
    'mvtec': load_mvtec,
    'mvtec2': load_mvtec2,
    'visa': load_visa,
    'mpdd': load_mpdd,
    'dagm': load_dagm,
    'btad': load_btad,
    'dtd': load_dtd,
    'br35h': load_br35h,
    'brainmri': load_brainmri,
    'brain_tumor_mri': load_brain_tumor_mri,
    'isic': load_isic,
    'clinicdb': load_clinicdb,
    'colondb': load_colondb,
}

dataset_classes = {
    'mvtec': mvtec_classes,
    'mvtec2': mvtec2_classes,
    'visa': visa_classes,
    'mpdd': mpdd_classes,
    'dagm': dagm_classes,
    'btad': btad_classes,
    'dtd': dtd_classes,
    'br35h': br35h_classes,
    'brainmri': brainmri_classes,
    'brain_tumor_mri': brain_tumor_mri_classes,
    'isic': isic_classes,
    'clinicdb': clinicdb_classes,
    'colondb': colondb_classes,
}

def denormalization(x):
    x = (((x.transpose(1, 2, 0) * std_train) + mean_train) * 255.).astype(np.uint8)
    return x

def get_dataloader_from_args(phase, **kwargs):

    dataset_inst = CLIPDataset(
        load_function=load_function_dict[kwargs['dataset']],
        category=kwargs['class_name'],
        phase=phase,
        k_shot=kwargs['k_shot'],
        experiment_indx=kwargs['experiment_indx']
    )

    if phase == 'train':
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=True,
                                  num_workers=0)
    else:
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=False,
                                 num_workers=0)


    debug_str = f"===> datasets: {kwargs['dataset']}, class name/len: {kwargs['class_name']}/{len(dataset_inst)}, batch size: {kwargs['batch_size']}"
    logger.info(debug_str)

    return data_loader, dataset_inst
