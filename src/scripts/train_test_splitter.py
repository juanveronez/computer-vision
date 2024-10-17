from image_transformer import CategorizedImageTransformerBuilder
from sklearn.model_selection import StratifiedShuffleSplit
import os
from pathlib import Path
import numpy as np

X, y = CategorizedImageTransformerBuilder() \
  .def_root_path(Path(os.path.join('..', 'data_raw'))) \
  .add_label('a', 'a_l/train_61') \
  .add_label('e', 'e_l/train_65') \
  .add_label('i', 'i_l/train_69') \
  .add_label('o', 'o_l/train_6f') \
  .add_label('u', 'u_l/train_75') \
  .add_label('A', 'A_u/train_41') \
  .add_label('E', 'E_u/train_45') \
  .add_label('I', 'I_u/train_49') \
  .add_label('O', 'O_u/train_4f') \
  .add_label('U', 'U_u/train_55') \
  .build(lambda y: (y == 'i') | (y == 'I'))

def train_test_split(X, y):
  split_test_threshold = 0.2
  selection_iter = StratifiedShuffleSplit(n_splits=1, test_size=split_test_threshold, random_state=42)

  train_index, test_index = next(selection_iter.split(X, y))

  X_train = X[train_index]
  X_test = X[test_index]
  y_train = y[train_index]
  y_test = y[test_index]

  return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y)

def save_file(file_name, data):
  packed_data = np.packbits(data)
  np.save(f'../data_processed/{file_name}.npy', packed_data)


save_file('X', X)
save_file('y', y)
save_file('X_train', X_train)
save_file('X_test', X_test)
save_file('y_train', y_train)
save_file('y_test', y_test)