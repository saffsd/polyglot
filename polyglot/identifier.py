"""
Implementation of the core multi-language identifier class.

Marco Lui, April 2013
"""
import bz2, base64
import numpy as np
import os

from cPickle import loads
from collections import defaultdict

import config

class MultiLanguageIdentifier(object):
  """
  LD feature space tokenizer based on a stripped-down version of
  the LanguageIdentifier class of langid.py
  """

  @classmethod
  def list_langs(cls, model):
    """
    List the languages supported by a pre-trained model.

    @param model model string or path to file containing model string
    @returns list of languages supported
    """
    if os.path.exists(model):
      with open(model) as f:
        langs = cls.__unpack(f.read())[0]
    else:
      langs = cls.__unpack(model)[0]

    return langs


  @classmethod
  def __unpack(cls, string):
    return loads(bz2.decompress(base64.b64decode(string)))

  @classmethod
  def default(cls, n_iters = config.N_ITERS, max_lang = config.MAX_LANG, thresh=config.THRESHOLD):
    import pkgutil
    nb_classes, nb_ptc, tk_nextmove, tk_output = cls.__unpack(pkgutil.get_data('polyglot','models/default'))
   
    return cls( nb_classes, nb_ptc, tk_nextmove, tk_output, n_iters, max_lang, thresh)

  @classmethod
  def from_modelstring(cls, string, *args, **kwargs):
    nb_classes, nb_ptc, tk_nextmove, tk_output = cls.__unpack(string)
   
    return cls( nb_classes, nb_ptc, tk_nextmove, tk_output, *args, **kwargs)

  @classmethod
  def from_modelpath(cls, path, *args, **kwargs):
    with open(path) as f:
      return cls.from_modelstring(f.read(), *args, **kwargs)

  def __init__(self, nb_classes, nb_ptc, tk_nextmove, tk_output, n_iters, max_lang, thresh):
    self.nb_classes = nb_classes
    self.nb_ptc = nb_ptc
    self.tk_nextmove = tk_nextmove
    self.tk_output = tk_output
    self.n_iters = n_iters
    self.max_lang = max_lang
    self.thresh = thresh

  def instance2fv(self, text):
    """
    Map an instance into the feature space of the trained model.
    """
    if isinstance(text, unicode):
      text = text.encode('utf8')

    arr = np.zeros((self.nb_ptc.shape[0],), dtype='uint32')

    # Convert the text to a sequence of ascii values
    ords = map(ord, text)

    # Count the number of times we enter each state
    state = 0
    statecount = defaultdict(int)
    for letter in ords:
      state = self.tk_nextmove[(state << 8) + letter]
      statecount[state] += 1

    # Update all the productions corresponding to the state
    for state in statecount:
      for index in self.tk_output.get(state, []):
        arr[index] += statecount[state]

    return arr

  def explain(self, fv, iters = None, alpha = 0., subset=None):
    """
    Explain a feature vector in terms of a set of classes.
    Uses a Gibbs sampler to compute the most likely class distribution
    over the specified class set to have generated this feature vector.

    @param subset specifies the subset of classes to use
    @returns counts of how many documents have been allocated to each topic
    """

    if iters is None:
      iters = self.n_iters

    if subset is None:
      ptc = self.nb_ptc
    else:
      ptc = self.nb_ptc[:,subset]

    # Initially random allocation of terms to topics
    K = ptc.shape[1] # number of topics (languages)
    z_n = np.random.randint(0, K, fv.sum())
    n_m_z = np.bincount(z_n, minlength=K)

    for i in range(iters):
        # We have a collased representation of the document, where we
        # only keep the counts of terms and not their relative ordering
        # (which the model assumes is fully exchangeable anyway)
        n = 0 # keep track of the feature index
        for t, n_t in enumerate(fv):
          for _ in xrange(n_t):
            # discount for n-th word t with topic z
            z = z_n[n]
            n_m_z[z] -= 1

            # sampling topic new_z for t
            # TODO: Can this be any faster?
            p_z = ptc[t] * (n_m_z + alpha) 
            new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()

            # set z the new topic and increment counters
            z_n[n] = new_z
            n_m_z[new_z] += 1

            n += 1

    # n_m_z must be projected back into the full class space
    retval = np.zeros((self.nb_ptc.shape[1],), dtype=int)
    retval[subset] = n_m_z

    return retval

  def logprob(self, fv, classes, iters=None, lam_c=None):
    """
    Compute the log-probability under our p(t|c) that the instance
    is composed by the given set of classes.
    """
    if lam_c is None:
      # most likely distribution assuming the set of classes
      lam_c = self.explain(fv, iters, subset=classes)
      lam_c = lam_c.astype(float) / lam_c.sum() # norm to 1

    #acc = len(classes) * np.log(1./len(fv)) # alternative prior
    acc = 0.
    for t, n_t in enumerate(fv):
      if n_t == 0: continue
      # there are n_t of token t in the document 
      # TODO: this should be a matrix product
      acc += n_t * np.log(sum(lam_c[c] * self.nb_ptc[t,c] for c in classes))
    return acc

  def score(self, text):
    # tokenize document into a distribution over terms
    fv = self.instance2fv(text) 

    dist = self.explain(fv)
    cl_order = np.arange(len(dist))[dist.argsort()]

    doclen = np.sum(fv)

    # Tabulate a list of logprob differences after adding each language in
    # the sequence. We normalize by dividing by the document length.
    logprobs = []
    for i in range(1,self.max_lang):
      cl_set = cl_order[-i:]
      est_lp = self.logprob(fv, cl_order[-i:], self.n_iters)
      logprobs.append(est_lp / doclen)

    retval = np.zeros(len(dist))
    retval[cl_order[-1]] = 5.0 #hardcode 5.0 for most likely class
    for v, l in zip( np.diff(logprobs), cl_order[-self.max_lang:-1][::-1] ):
      retval[l] = v

    return retval

  def scoremap(self, text):
    score = self.score(text)
    assert len(score) == len(self.nb_classes)
    retval = dict(zip(self.nb_classes, score))
    return retval

  def identify(self, text):
    score = self.score(text)
    
    langs = self.nb_classes
    retval = []
    for v,l in zip(score, langs):
      if v > self.thresh:
        retval.append(l)

    return retval
