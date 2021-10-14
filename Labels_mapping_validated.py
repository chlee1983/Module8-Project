import os

train_dir = 'train'


def find_classes(train_dir):
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(train_dir) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {train_dir}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


mapping_result = find_classes(train_dir)

x, y = mapping_result

for key, values in y.items():
    print('class label is: ', key, ' internal label is: ', values)
