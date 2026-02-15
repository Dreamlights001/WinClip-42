import glob
import os
import random

isic_classes = ['isic2016']

ISIC_DIR = './datasets/ISBI2016'


def load_isic(category, k_shot, experiment_indx):
    def load_phase(img_path, gt_path=None, is_train=False):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if not os.path.exists(img_path):
            return img_tot_paths, gt_tot_paths, tot_labels, tot_types

        img_paths = glob.glob(os.path.join(img_path) + "/*.png")
        img_paths.extend(glob.glob(os.path.join(img_path) + "/*.jpg"))
        img_paths.extend(glob.glob(os.path.join(img_path) + "/*.JPG"))
        img_paths.sort()

        for img_path_item in img_paths:
            img_name = os.path.basename(img_path_item)
            img_name_no_ext = img_name.rsplit('.', 1)[0]

            if gt_path is not None and os.path.exists(gt_path):
                gt_files = glob.glob(os.path.join(gt_path, img_name_no_ext + ".*"))
                if gt_files:
                    gt_path_item = gt_files[0]
                else:
                    gt_path_item = 0
            else:
                gt_path_item = 0

            img_tot_paths.append(img_path_item)
            gt_tot_paths.append(gt_path_item)

            if gt_path_item != 0 and os.path.exists(gt_path_item):
                tot_labels.append(1)
                tot_types.append('lesion')
            else:
                tot_labels.append(0)
                tot_types.append(['normal'] if is_train else ['lesion'])

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in isic_classes
    assert k_shot in [0, 1, 5, 10]
    assert experiment_indx in [0, 1, 2]

    train_img_path = os.path.join(ISIC_DIR, 'ISBI2016_ISIC_Part1_Training_Data')
    train_gt_path = os.path.join(ISIC_DIR, 'ISBI2016_ISIC_Part1_Training_GroundTruth')
    test_img_path = os.path.join(ISIC_DIR, 'ISBI2016_ISIC_Part1_Test_Data')
    test_gt_path = os.path.join(ISIC_DIR, 'ISBI2016_ISIC_Part1_Test_GroundTruth')

    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, \
    train_tot_types = load_phase(train_img_path, train_gt_path, is_train=True)

    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, \
    test_tot_types = load_phase(test_img_path, test_gt_path, is_train=False)

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
