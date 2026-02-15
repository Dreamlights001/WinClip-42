import glob
import os
import random

visa_classes = ['candle', 'capsules', 'cashew', 'chewinggum',
                   'fryum', 'macaroni1', 'macaroni2',
                'pcb1', 'pcb2', 'pcb3','pcb4', 'pipe_fryum']

VISA_DIR = '../datasets/visa'


def load_visa(category, k_shot, experiment_indx):
    def load_phase(img_path, mask_path=None):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if not os.path.exists(img_path):
            return img_tot_paths, gt_tot_paths, tot_labels, tot_types

        for label_type in ['Normal', 'Anomaly']:
            label_path = os.path.join(img_path, label_type)
            if not os.path.exists(label_path):
                continue

            img_paths = glob.glob(os.path.join(label_path) + "/*.png")
            img_paths.extend(glob.glob(os.path.join(label_path) + "/*.jpg"))
            img_paths.extend(glob.glob(os.path.join(label_path) + "/*.JPG"))
            img_paths.extend(glob.glob(os.path.join(label_path) + "/*.jpeg"))

            if label_type == 'Normal':
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_tot_paths.extend(img_paths)
                if mask_path is not None and os.path.exists(mask_path):
                    for img_path_item in img_paths:
                        img_name = os.path.basename(img_path_item)
                        img_name_no_ext = img_name.rsplit('.', 1)[0]
                        gt_files = glob.glob(os.path.join(mask_path, img_name_no_ext + ".*"))
                        if not gt_files:
                            gt_files = glob.glob(os.path.join(mask_path, img_name))
                        if gt_files:
                            gt_tot_paths.append(gt_files[0])
                        else:
                            gt_tot_paths.append(0)
                else:
                    gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend(['anomaly'] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in visa_classes
    assert k_shot in [0, 1, 5, 10]
    assert experiment_indx in [0, 1, 2]

    img_path = os.path.join(VISA_DIR, category, 'Data', 'Images')
    mask_path = os.path.join(VISA_DIR, category, 'Data', 'Masks', 'Anomaly')

    all_img_tot_paths, all_gt_tot_paths, all_tot_labels, all_tot_types = load_phase(img_path, mask_path)

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

    if k_shot > 0:
        random.seed(42)
        full_index = list(range(len(train_img_tot_paths)))
        selected_index = random.sample(full_index, min(k_shot, len(full_index)))
        selected_train_img_tot_paths = [train_img_tot_paths[k] for k in selected_index]
        selected_train_gt_tot_paths = [train_gt_tot_paths[k] for k in selected_index]
        selected_train_tot_labels = [train_tot_labels[k] for k in selected_index]
        selected_train_tot_types = [train_tot_types[k] for k in selected_index]

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels, selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)
