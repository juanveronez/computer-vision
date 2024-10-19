from image_transformer import CategorizedImageTransformerBuilder

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

def save_file(file_name, data):
  np.save(f'../data_processed/{file_name}.npy', data)

save_file('X', X)
save_file('y', y)
