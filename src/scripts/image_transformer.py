from cv2 import imread, IMREAD_GRAYSCALE
import numpy as np
import os
from pathlib import Path

class CategorizedImageTransformerBuilder:
  path_labels = {}
  label_to_file = {}

  def def_root_path(self, root_path: Path):
    self.root_path = root_path
    return self
  
  def add_label(self, label: str, relative_path: str):
    if label in self.path_labels:
      raise ValueError(f"Label {label} already exists in the path_labels")
    
    if not hasattr(self, 'root_path'):
      raise ValueError("Root path is not defined")
    
    self.path_labels[label] = self.root_path.joinpath(relative_path)
    return self

  def __transform_label(self):
    self.label_to_file = {
      k: [path_label / file for file in os.listdir(path_label)]
      for k, path_label in self.path_labels.items()
    }

  def __transform_image(self, target_func):
    X = []
    y = []
    for key, files in self.label_to_file.items():
        for path in files:
            y.append(key)
            image = imread(path, IMREAD_GRAYSCALE)
            binary_matrix = (image > 0).astype(np.int8)
            X.append(binary_matrix)

    X = np.array([x.reshape(-1) for x in X])
    y = target_func(np.array(y))

    return X, y

  def build(self, target_func):
    if not hasattr(self, 'root_path') or len(self.path_labels) == 0:
      raise ValueError("Root path is not defined or path labels are empty")

    self.__transform_label()
    return self.__transform_image(target_func)
