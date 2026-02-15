import glob
import os
import random

br35h_classes = ['br35h']
brainmri_classes = ['brainmri']
brain_tumor_mri_classes = ['brain_tumor_mri']

BR35H_DIR = './datasets/Br35H'
BRAINMRI_DIR = './datasets/BrainMRI'
BRAIN_TUMOR_MRI_DIR = './datasets/Brain Tumor MRI Dataset'


def load_br35h(category, k_shot, experiment_indx):
    def load_phase(root_path, gt_path=None):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if not os.path.exists(root_path):
            return img_tot_paths, gt_tot_paths, tot_labels, tot_types

        for label_type in ['no', 'yes']:
            label_path = os.path.join(root_path, label_type)
            if not os.path.exists(label_path):
                continue

            img_paths = glob.glob(os.path.join(label_path) + "/*.png")
            img_paths.extend(glob.glob(os.path.join(label_path) + "/*.jpg"))
            img_paths.extend(glob.glob(os.path.join(label_path) + "/*.JPG"))
            img_paths.extend(glob.glob(os.path.join(label_path) + "/*.jpeg"))

            if label_type == 'no':
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend(['tumor'] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in br35h_classes
    assert k_shot in [0, 1, 5, 10]
    assert experiment_indx in [0, 1, 2]

    train_img_path = os.path.join(BR35H_DIR, 'TRAIN')
    test_img_path = os.path.join(BR35H_DIR, 'TEST')

    if not os.path.exists(train_img_path):
        train_img_path = os.path.join(BR35H_DIR, 'yes')
        all_img_tot_paths, all_gt_tot_paths, all_tot_labels, all_tot_types = load_phase(BR35H_DIR)
        all_img_tot_paths.extend(glob.glob(os.path.join(BR35H_DIR, 'no') + "/*.png"))
        all_img_tot_paths.extend(glob.glob(os.path.join(BR35H_DIR, 'no') + "/*.jpg"))

        random.seed(42)
        random.shuffle(all_img_tot_paths)

        split_idx = int(len(all_img_tot_paths) * 0.8)
        train_img_tot_paths = all_img_tot_paths[:split_idx]
        train_gt_tot_paths = all_gt_tot_paths[:split_idx]
        train_tot_labels = all_tot_labels[:split_idx]
        train_tot_types = all_tot_types[:split_idx]

        test_img_tot_paths = all_img_tot_paths[split_idx:]
        test_gt_tot_paths = all_gt_tot_paths[split_idx:]
        test_tot_labels = all_tot_labels[split_idx:]
        test_tot_types = all_tot_types[split_idx:]
    else:
        train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types = load_phase(train_img_path)
        test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types = load_phase(test_img_path)

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


def load_brainmri(category, k_shot, experiment_indx):
    def load_phase(root_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if not os.path.exists(root_path):
            return img_tot_paths, gt_tot_paths, tot_labels, tot_types

        for label_type in ['no', 'yes']:
            label_path = os.path.join(root_path, label_type)
            if not os.path.exists(label_path):
                label_path = os.path.join(root_path, 'brain_tumor_dataset', label_type)
            if not os.path.exists(label_path):
                continue

            img_paths = glob.glob(os.path.join(label_path) + "/*.png")
            img_paths.extend(glob.glob(os.path.join(label_path) + "/*.jpg"))
            img_paths.extend(glob.glob(os.path.join(label_path) + "/*.JPG"))
            img_paths.extend(glob.glob(os.path.join(label_path) + "/*.jpeg"))

            if label_type == 'no':
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend(['tumor'] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in brainmri_classes
    assert k_shot in [0, 1, 5, 10]
    assert experiment_indx in [0, 1, 2]

    all_img_tot_paths, all_gt_tot_paths, all_tot_labels, all_tot_types = load_phase(BRAINMRI_DIR)

    random.seed(42)
    combined = list(zip(all_img_tot_paths, all_gt_tot_paths, all_tot_labels, all_tot_types))
    random.shuffle(combined)
    all_img_tot_paths, all_gt_tot_paths, all_tot_labels, all_tot_types = zip(*combined)
    all_img_tot_paths = list(all_img_tot_paths)
    all_gt_tot_paths = list(all_gt_tot_paths)
    all_tot_labels = list(all_tot_labels)
    all_tot_types = list(all_tot_types)

    split_idx = int(len(all_img_tot_paths) * 0.8)
    train_img_tot_paths = all_img_tot_paths[:split_idx]
    train_gt_tot_paths = all_gt_tot_paths[:split_idx]
    train_tot_labels = all_tot_labels[:split_idx]
    train_tot_types = all_tot_types[:split_idx]

    test_img_tot_paths = all_img_tot_paths[split_idx:]
    test_gt_tot_paths = all_gt_tot_paths[split_idx:]
    test_tot_labels = all_tot_labels[split_idx:]
    test_tot_types = all_tot_types[split_idx:]

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


def load_brain_tumor_mri(category, k_shot, experiment_indx):
    def load_phase(root_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if not os.path.exists(root_path):
            return img_tot_paths, gt_tot_paths, tot_labels, tot_types

        tumor_types = os.listdir(root_path)

        for tumor_type in tumor_types:
            tumor_path = os.path.join(root_path, tumor_type)
            if not os.path.isdir(tumor_path):
                continue

            img_paths = glob.glob(os.path.join(tumor_path) + "/*.png")
            img_paths.extend(glob.glob(os.path.join(tumor_path) + "/*.jpg"))
            img_paths.extend(glob.glob(os.path.join(tumor_path) + "/*.JPG"))
            img_paths.extend(glob.glob(os.path.join(tumor_path) + "/*.jpeg"))

            if tumor_type == 'notumor':
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([tumor_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in brain_tumor_mri_classes
    assert k_shot in [0, 1, 5, 10]
    assert experiment_indx in [0, 1, 2]

    train_img_path = os.path.join(BRAIN_TUMOR_MRI_DIR, 'Training')
    test_img_path = os.path.join(BRAIN_TUMOR_MRI_DIR, 'Testing')

    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types = load_phase(train_img_path)
    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types = load_phase(test_img_path)

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
