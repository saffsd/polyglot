"""
Convert langid.py model to one suitable for use in multilangid.py.
The main issue is that we need to renormalize P(t|C) as it is stored as
log-prob in langid.py.

Marco Lui, March 2013
"""
import argparse, os
import numpy as np

import bz2, base64
from cPickle import loads, dumps

def read_nb_model(path):
  def model_file(name):
    return os.path.join(path, name)

  with open(model_file('model')) as f:
    model = loads(bz2.decompress(base64.b64decode(f.read())))
  nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output = model
  nb_numfeats = len(nb_ptc) / len(nb_pc)
  nb_ptc = np.array(nb_ptc).reshape(len(nb_ptc)/len(nb_pc), len(nb_pc))

  # Normalize to 1 on the term axis
  for i in range(nb_ptc.shape[1]):
    nb_ptc[:,i] = (1/np.exp(nb_ptc[:,i][None,:] - nb_ptc[:,i][:,None]).sum(1))

  return (nb_classes, nb_ptc, tk_nextmove, tk_output)



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('model', metavar="MODEL_DIR", help="path to langid.py training model dir")
  parser.add_argument('output', metavar="OUTPUT", help="produce output in")
  args = parser.parse_args()
  
  model = read_nb_model(args.model)
  output = base64.b64encode(bz2.compress(dumps(model)))
  with open(args.output, 'w') as f:
    f.write(output)

if __name__ == "__main__":
  main()
