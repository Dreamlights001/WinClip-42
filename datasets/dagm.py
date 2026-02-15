import glob
import os
import random

dagm_classes = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5',
                'Class6', 'Class7', 'Class8', 'Class9', 'Class10']

DAGM_DIR = './datasets/dagm'


def load_dagm(category, k_shot, experiment_indx):
    def load_phase(img_root_path, label_root_path, is_train=False):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if not os.path.exists(img_root_path):
            return img_tot_paths, gt_tot_paths, tot_labels, tot_types

        img_paths = glob.glob(os.path.join(img_root_path) + "/*.png")
        img_paths.extend(glob.glob(os.path.join(img_root_path) + "/*.PNG"))
        img_paths.extend(glob.glob(os.path.join(img_root_path) + "/*.jpg"))
        img_paths.extend(glob.glob(os.path.join(img_root_path) + "/*.JPG"))
        img_paths.sort()

        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            img_name_no_ext = img_name.rsplit('.', 1)[0]

            if label_root_path is not None and os.path.exists(label_root_path):
                label_path = glob.glob(os.path.join(label_root_path, img_name_no_ext + ".*"))
                if label_path:
                    label_path = label_path[0]
                else:
                    label_path = 0
            else:
                label_path = 0

            img_tot_paths.append(img_path)
            gt_tot_paths.append(label_path)

            if is_train:
                tot_labels.append(0)
                tot_types.append('good')
            else:
                if label_path != 0 and os.path.exists(label_path):
                    tot_labels.append(1)
                    tot_types.append('defect')
                else:
                    tot_labels.append(0)
                    tot_types.append('good')

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in dagm_classes
    assert k_shot in [0, 1, 5, 10]
    assert experiment_indx in [0, 1, 2]

    train_img_path = os.path.join(DAGM_DIR, category, 'Train')
    train_label_path = os.path.join(DAGM_DIR, category, 'Train', 'Label')
    test_img_path = os.path.join(DAGM_DIR, category, 'Test')
    test_label_path = os.path.join(DAGM_DIR, category, 'Test', 'Label')

    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, \
    train_tot_types = load_phase(train_img_path, train_label_path, is_train=True)

    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, \
    test_tot_types = load_phase(test_img_path, test_label_path, is_train=False)

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
