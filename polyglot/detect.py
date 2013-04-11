"""
Multi-langid based on a pre-trained P(w|t) and a Gibbs sampler for
estimating P(t|d).

Marco Lui, March 2013
"""
import argparse, sys, csv
import multiprocessing as mp
import numpy as np
import logging
logger = logging.getLogger(__name__)


from identifier import MultiLanguageIdentifier
from utils import Timer, MapPool
import config

def setup_identify(model_path, langs=None, n_iters=None, max_lang=None, thresh=None, prior=None):
  global _identifier

  n_iters = n_iters if n_iters is not None else config.N_ITERS
  max_lang = max_lang if max_lang is not None else config.MAX_LANG
  thresh = thresh if thresh is not None else config.THRESHOLD
  _identifier = MultiLanguageIdentifier.from_modelpath(model_path, langs, n_iters, max_lang, thresh, prior)

def setup_default_identify(langs=None, n_iters = None, max_lang=None, thresh=None, prior=None):
  global _identifier

  n_iters = n_iters if n_iters is not None else config.N_ITERS
  max_lang = max_lang if max_lang is not None else config.MAX_LANG
  thresh = thresh if thresh is not None else config.THRESHOLD
  _identifier = MultiLanguageIdentifier.default(langs, n_iters, max_lang, thresh, prior)


def explain(path):
  """
  Explain the document as a distribution of tokens over the full language set.
  """
  global _identifier

  with open(path) as f:
    fv = _identifier.instance2fv(f.read())
    retval = _identifier.explain(fv)

  return [path] + list(retval)

def identify(path):
  global _identifier

  with open(path) as f:
    text = f.read()
    try:
      pred = _identifier.identify(text)
    except ValueError:
      pred = []

  return [path] + pred

def tokenize(path):
  global _identifier

  with open(path) as f:
    text = f.read()

  return _identifier.instance2fv(text)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--iters','-i',type=int, metavar='N', help="perform N iterations of Gibbs sampling", default=config.N_ITERS)
  parser.add_argument('--jobs','-j',type=int, metavar='N', help="use N processes", default=mp.cpu_count())
  parser.add_argument('--output','-o', help="output file (csv format)")
  parser.add_argument('--max_lang', type=int, help="maximum number of langugages to consider per-document", default=config.MAX_LANG)
  parser.add_argument('--thresh', '-t', type=float, help="threshold for including a language", default=config.THRESHOLD)
  parser.add_argument('--model', '-m', metavar="MODEL", help="path to model")
  parser.add_argument('--verbose', '-v', action='store_true', help="verbose output")
  parser.add_argument('--explain', '-e', action='store_true', help="only explain documents as a breakdown over the full language set")
  parser.add_argument('-l', '--langs', dest='langs', help='comma-separated set of target ISO639 language codes (e.g en,de)')
  parser.add_argument('--prior', '-p', nargs="?", const=True, help="use prior from file PRIOR (computed if PRIOR is not specified)")

  parser.add_argument('docs', metavar='FILE', help='file to process (read from stdin if blank)', nargs='*')

  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

  if args.output:
    out_f = open(args.output, 'w')
  else:
    out_f = sys.stdout
  writer = csv.writer(out_f)

  if args.docs:
    doclist = args.docs
  else:
    doclist = map(str.strip, sys.stdin)

  chunksize = max(1,len(doclist) / (args.jobs + 4))

  logger.info( "processing {0} docs".format(len(doclist)) )
  
  if args.langs:
    langs = args.langs.strip().split(',')
    logger.debug( "restricting language set to: {0}".format(langs))
  else:
    langs = None

  if args.model:
    initalizer = setup_identify
    initargs = (args.model, langs, args.iters, args.max_lang, args.thresh)
    langs = list(MultiLanguageIdentifier.list_langs(args.model))
  else:
    initalizer = setup_default_identify
    initargs = (langs, args.iters, args.max_lang, args.thresh)
    langs = list(MultiLanguageIdentifier.list_langs())

  if args.prior:
    if args.prior is True:
      logger.debug("using average document as prior")
      with MapPool(args.jobs, initalizer, initargs, chunksize=chunksize) as p:
        fvs = [ v.astype(float) / v.sum() for v in p(tokenize, doclist)]
      prior = np.sum(fvs, axis=0)
    else:
      logger.debug("loading prior from: {0}".format(args.prior))
      with open(args.prior) as f:
        reader = csv.reader(f)
        prior = map(float, reader.next())

    initargs += (prior,)

  # Determine the type of output
  if args.explain:
    process = explain 
    writer.writerow(['path'] + langs)
  else:
    process = identify


  # Process the documents specified
  doc_count = 0
  with MapPool(args.jobs, initalizer, initargs, chunksize=chunksize) as p, Timer() as t:
    for retval in p(process, doclist):
      writer.writerow( retval )
      doc_count += 1
      logger.info("processed {0} docs in {1:.2f}s ({2:.2f} r/s)".format(doc_count, t.elapsed, t.rate(doc_count) ))

if __name__ == "__main__":
  main()
