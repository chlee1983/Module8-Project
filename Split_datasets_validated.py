import math
import shutil
import os


def split_dataset_into_3(path_to_dataset, derived_path, train_ratio, valid_ratio):
    """
    split the dataset in the given path into three subsets(test,validation,train)
    :param path_to_dataset:
    :param train_ratio:
    :param valid_ratio:
    :return:
    """

    """sub_dirs will retrieve the name of subdirectories, 
    path_to_dataset is the directory where the original datasets are stored"""
    _, sub_dirs, _ = next(iter(os.walk(path_to_dataset)))  # retrieve name of subdirectories
    sub_dir_item_cnt = [0 for i in range(len(sub_dirs))]  # list for counting items in each sub directory(class)

    """directories where the splitted dataset will lie,
    it will create the sub-directory under ./label with the name 'train', 'validation' etc"""
    dir_train = os.path.join(os.path.dirname(derived_path), 'train')
    dir_valid = os.path.join(os.path.dirname(derived_path), 'validation')
    dir_test = os.path.join(os.path.dirname(derived_path), 'test')

    """sub_dir is the sub-directory 0, 1, 2... including train and validation"""
    for i, sub_dir in enumerate(sub_dirs):
        """print sub-directory 0, 1, 2... inside train and validation directory"""
        dir_train_dst = os.path.join(dir_train, sub_dir)  # directory for destination of train dataset
        dir_valid_dst = os.path.join(dir_valid, sub_dir)  # directory for destination of validation dataset
        dir_test_dst = os.path.join(dir_test, sub_dir)  # directory for destination of test dataset

        # variables to save the sub directory name(class name) and to count the images of each sub directory(class)
        class_name = sub_dir
        sub_dir = os.path.join(path_to_dataset, sub_dir)
        """the following will count the length of (sub_dir)"""
        sub_dir_item_cnt[i] = len(os.listdir(sub_dir))
        """items will list out the content of the sub-dir (i.e. in jpg format)"""
        items = os.listdir(sub_dir)
        print(sub_dir + " total count inside sub-directory is " + str(sub_dir_item_cnt[i]))

        # transfer data to trainset
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio)):
            if not os.path.exists(dir_train_dst):
                os.makedirs(dir_train_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_train_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # transfer data to validation
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio),
                              round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio))):
            if not os.path.exists(dir_valid_dst):
                os.makedirs(dir_valid_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_valid_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # transfer data to testset
        for item_idx in range(round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio)), sub_dir_item_cnt[i]):
            if not os.path.exists(dir_test_dst):
                os.makedirs(dir_test_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_test_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        for item_idx in range(round(sub_dir_item_cnt[i])):
            if not os.path.exists(dir_valid_dst):
                os.makedirs(dir_valid_dst)
                source_file = os.path.join(sub_dir, items[item_idx])
                dst_file = os.path.join(dir_valid_dst, items[item_idx])
                shutil.copyfile(source_file, dst_file)
    return


original_dataset_path = "./label/"
derived_dataset_path = "./Module8 Project"
tr_ratio = 0.9
vl_ratio = 0.1
split_dataset_into_3(original_dataset_path, derived_dataset_path, tr_ratio, vl_ratio)
