"""
Implementation of the core multi-language identifier class.

Marco Lui, April 2013
"""
import bz2, base64
import numpy as np
import os
import pkgutil
import logging

logger = logging.getLogger(__name__)

from cPickle import loads
from collections import defaultdict

import config
from itertools import compress

class MultiLanguageIdentifier(object):
  """
  LD feature space tokenizer based on a stripped-down version of
  the LanguageIdentifier class of langid.py
  """

  @classmethod
  def list_langs(cls, model=None):
    """
    List the languages supported by a pre-trained model.

    @param model model string or path to file containing model string
    @returns list of languages supported
    """
    if model is None:
      langs = cls.unpack_model(pkgutil.get_data('polyglot','models/default'))[0]
    elif os.path.exists(model):
      with open(model) as f:
        langs = cls.unpack_model(f.read())[0]
    else:
      langs = cls.unpack_model(model)[0]

    return langs


  @classmethod
  def unpack_model(cls, string):
    return loads(bz2.decompress(base64.b64decode(string)))

  @classmethod
  def default(cls, *args, **kwargs):
    nb_classes, nb_ptc, tk_nextmove, tk_output = cls.unpack_model(pkgutil.get_data('polyglot','models/default'))
   
    return cls( nb_classes, nb_ptc, tk_nextmove, tk_output, *args, **kwargs)

  @classmethod
  def from_modelstring(cls, string, *args, **kwargs):
    nb_classes, nb_ptc, tk_nextmove, tk_output = cls.unpack_model(string)
   
    return cls( nb_classes, nb_ptc, tk_nextmove, tk_output, *args, **kwargs)

  @classmethod
  def from_modelpath(cls, path, *args, **kwargs):
    with open(path) as f:
      return cls.from_modelstring(f.read(), *args, **kwargs)

  def __init__(self, nb_classes, nb_ptc, tk_nextmove, tk_output, langs, n_iters, max_lang, thresh, prior):
    self.tk_nextmove = tk_nextmove
    self.tk_output = tk_output
    self.n_iters = n_iters
    self.max_lang = max_lang
    self.thresh = thresh

    # Class 0 is used for the prior over the feature set
    if langs is None:
      self.nb_classes = ('PRIOR',) + tuple(nb_classes)
    else:
      self.nb_classes = ('PRIOR',) + tuple(langs) 

    # Prepare prior and attach it to nb_ptc
    if prior is None:
      prior = np.ones(nb_ptc.shape[0])
    elif len(prior) != nb_ptc.shape[0]:
      raise ValueError("length of prior does not match number of terms in ptc")
    prior = np.array(prior, dtype=float) / np.sum(prior) # Normalize to sum 1

    if langs is None:
      self.nb_ptc = np.hstack((prior[:,None], nb_ptc))
    else:
      self.nb_ptc = np.hstack((prior[:,None], nb_ptc[:,[nb_classes.index(l) for l in langs]]))

    logger.debug("initialized a MultiLanguageIdentifier instance")
    logger.debug("n_iters: {0}".format(self.n_iters))
    logger.debug("max_lang: {0}".format(self.max_lang))
    logger.debug("thresh: {0}".format(self.thresh))
    logger.debug("ptc shape: {0}".format(self.nb_ptc.shape))

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

  def explain(self, fv, iters = None, alpha = 0., subset = None):
    """
    Explain a feature vector in terms of a set of classes.
    Uses a Gibbs sampler to compute the most likely class distribution
    over the specified class set to have generated this feature vector.

    @param subset specifies the subset of classes to use (defaults to all)
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
    n_m_z = np.bincount(z_n, minlength=K) + alpha

    t_nz = list(compress(enumerate(fv), fv>0))

    for i in range(iters):
        # We have a collased representation of the document, where we
        # only keep the counts of terms and not their relative ordering
        # (which the model assumes is fully exchangeable anyway)
        n = 0 # keep track of the feature index
        for t, n_t in t_nz:
          for _ in xrange(n_t):
            # discount for n-th word t with topic z
            z = z_n[n]
            n_m_z[z] -= 1

            # sampling topic new_z for t
            dist = np.cumsum(ptc[t] * n_m_z)
            samp = np.random.random() * dist[-1]
            new_z = np.searchsorted(dist,samp)

            # set z the new topic and increment counters
            z_n[n] = new_z
            n_m_z[new_z] += 1

            n += 1

    # n_m_z must be projected back into the full class space
    retval = np.zeros((self.nb_ptc.shape[1],), dtype=int)
    retval[subset] = (n_m_z - alpha).astype(int)

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

    nz_t = fv > 0 # non-zero features
    prod = lam_c[classes] * self.nb_ptc[:,classes][nz_t] 
    acc = np.sum(fv[nz_t] * np.log(np.sum(prod, axis=1)))
    return acc

  def identify(self, text):
    # tokenize document into a distribution over terms
    fv = self.instance2fv(text) 
    doclen = np.sum(fv)
    if doclen == 0:
      # no LD tokens -> no languages present
      return {}

    dist = self.explain(fv)
    logger.debug("prior: {0} / {1} ({2:.1f}%)".format(dist[0], dist.sum(), dist[0]*100. / dist.sum()))
    cl_order = np.arange(len(dist))[dist.argsort()][::-1]

    # initially explain the document only in terms of the prior
    lp = self.logprob(fv, [0])
    cl_set = [0]
    cl_dist = np.array([1.])

    for new_cl in [c for c in cl_order if c != 0 ][:self.max_lang]:
      cl_set_n = cl_set + [new_cl]
      # We obtain lam_c distinct from logprob as we will need it if we decide to keep.
      lam_c = self.explain(fv, subset=cl_set_n)
      lam_c = lam_c.astype(float) / lam_c.sum() # norm to 1
      est_lp = self.logprob(fv, cl_set_n, lam_c=lam_c)
      improve = (est_lp - lp) / doclen
      logger.debug("  {0} improves by {1:.3f}".format(self.nb_classes[new_cl], improve))
      if improve > self.thresh:
        lp = est_lp
        cl_set = cl_set_n
        cl_dist = lam_c

    # Re-normalize the mass over the languages to 1 - ignoring the class0 mass.
    cl_dist[1:] /= cl_dist[1:].sum()

    retval = { self.nb_classes[c]:cl_dist[c] for c in cl_set[1:]}
    return retval
