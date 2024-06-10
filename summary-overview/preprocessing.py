# split univariate seq into samples for supervised learning - For a simple MLP model
def split_seq(seq,n_steps):
  '''Given a sequence and number of steps, generates X, y (samples) for a simple MLP model
  in:
    seq: sequence of univariate variables
    n_step: number of steps
  out:
    array(X), array(y)    
  '''
  X, y = list(), list()
  for i in range (len(seq)):
    last_element = i + n_steps
    if last_element > len(seq) - 1:
      break
    else: 
      seq_x , seq_y = seq[i:last_element]. seq[last_element]
      X.append(seq_x)
      y.append(seq_y)
  return array(X), array(y)


def simple_to_3d(X, time_step):
  '''Given a 2D X (samples, features) return 3D X (time_step, samples, features)
  '''
  return X.reshape(time_step, -1)
