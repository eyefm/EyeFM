import os
import os.path
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.jpx')


def pil_loader(path: str, convert_rgb=True) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # with open(path, 'rb') as f:
    #     img = Image.open(f)
    img = Image.open(path)
    return img.convert('RGB') if convert_rgb else img


class MultiTaskDatasetFolderFromCSV(VisionDataset):
    def __init__(
            self,
            root: str,
            tasks: List[str],
            csv_data,
            nb_classes: int,
            label_column: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            prefixes: Optional[Dict[str, str]] = None,
            max_images: Optional[int] = None
    ) -> None:
        super(MultiTaskDatasetFolderFromCSV, self).__init__(root, transform=transform,
                                                     target_transform=target_transform)
        self.tasks = tasks
        # classes, class_to_idx = self._find_classes(os.path.join(self.root, self.tasks[0]))
        classes = [f'{i}' for i in range(nb_classes)]
        class_to_idx = {f'{i}': i for i in range(nb_classes)}

        prefixes = {} if prefixes is None else prefixes
        prefixes.update({task: '' for task in tasks if task not in prefixes})

        samples = dict()
        for task in self.tasks:
            samples[task] = csv_data.apply(lambda row: (os.path.join(self.root, row[task]), row[label_column]), axis=1).tolist()

        for task, task_samples in samples.items():
            if len(task_samples) == 0:
                msg = "Found 0 logs in subfolders of: {}\n".format(os.path.join(self.root, task))
                if extensions is not None:
                    msg += "Supported extensions are: {}".format(",".join(extensions))
                raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        # Select random subset of dataset if so specified
        if isinstance(max_images, int):
            total_samples = len(list(self.samples.values())[0])
            np.random.seed(0)
            permutation = np.random.permutation(total_samples)
            for task in samples:
                self.samples[task] = [self.samples[task][i] for i in permutation][:max_images]

        self.cache = {}

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # target = None
        if index in self.cache:
            sample_dict, target = deepcopy(self.cache[index])
        else:
            sample_dict = {}
            for task in self.tasks:
                path, target = self.samples[task][index]
                sample = pil_loader(path, convert_rgb=True)
                sample_dict[task] = sample

        if self.transform is not None:
            for task in self.tasks:
                sample_dict[task] = self.transform(sample_dict[task])
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample_dict, target

    def __len__(self) -> int:
        return len(list(self.samples.values())[0])


class MultiTaskImageFolderFromCSV(MultiTaskDatasetFolderFromCSV):
    def __init__(
            self,
            root: str,
            tasks: List[str],
            csv_data,
            nb_classes: int,
            label_column: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            prefixes: Optional[Dict[str,str]] = None,
            max_images: Optional[int] = None
    ):
        super(MultiTaskImageFolderFromCSV, self).__init__(root, tasks, csv_data, nb_classes, label_column, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          prefixes=prefixes,
                                          max_images=max_images)
        self.imgs = self.samples