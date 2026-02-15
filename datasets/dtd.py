import glob
import os
import random

dtd_classes = ['Blotchy_099', 'Fibrous_183', 'Marbled_078', 'Matted_069', 'Mesh_114',
               'Perforated_037', 'Stratified_154', 'Woven_001', 'Woven_068', 'Woven_104',
               'Woven_125', 'Woven_127']

DTD_DIR = './datasets/DTD'


def load_dtd(category, k_shot, experiment_indx):
    def load_phase(root_path, gt_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if not os.path.exists(root_path):
            return img_tot_paths, gt_tot_paths, tot_labels, tot_types

        defect_types = os.listdir(root_path)

        for defect_type in defect_types:
            defect_path = os.path.join(root_path, defect_type)
            if not os.path.isdir(defect_path):
                continue

            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(defect_path) + "/*.png")
                img_paths.extend(glob.glob(os.path.join(defect_path) + "/*.jpg"))
                img_paths.extend(glob.glob(os.path.join(defect_path) + "/*.JPG"))
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(defect_path) + "/*.png")
                img_paths.extend(glob.glob(os.path.join(defect_path) + "/*.jpg"))
                img_paths.extend(glob.glob(os.path.join(defect_path) + "/*.JPG"))
                if gt_path is not None:
                    gt_paths = [os.path.join(gt_path, defect_type, os.path.basename(s).rsplit('.', 1)[0] + '.png') for s in
                                img_paths]
                else:
                    gt_paths = [0] * len(img_paths)
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in dtd_classes
    assert k_shot in [0, 1, 5, 10]
    assert experiment_indx in [0, 1, 2]

    test_img_path = os.path.join(DTD_DIR, category, 'test')
    train_img_path = os.path.join(DTD_DIR, category, 'train')
    ground_truth_path = os.path.join(DTD_DIR, category, 'ground_truth')

    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, \
    train_tot_types = load_phase(train_img_path, None)

    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, \
    test_tot_types = load_phase(test_img_path, ground_truth_path)

    selected_train_img_tot_paths = []
    selected_train_gt_tot_paths = []
    selected_train_tot_labels = []
    selected_train_tot_types = []

    if k_shot > 0 and len(train_img_tot_paths) > 0:
        random.seed(42)
        full_index = list(range(len(train_img_tot_paths)))
        selected_index = random.sample(full_index, min(k_shot, len(full_index)))
        selected_train_img_tot_paths = [train_img_tot_paths[k] for k in selected_index]
        selected_train_gt_tot_paths = [train_gt_tot_paths[k] for k in selected_index]
        selected_train_tot_labels = [train_tot_labels[k] for k in selected_index]
        selected_train_tot_types = [train_tot_types[k] for k in selected_index]

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels, selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)
