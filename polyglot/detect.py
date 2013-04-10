"""
Multi-langid based on a pre-trained P(w|t) and a Gibbs sampler for
estimating P(t|d).

Marco Lui, March 2013
"""
import argparse, sys, csv
import multiprocessing as mp

from identifier import MultiLanguageIdentifier
from utils import Timer, MapPool
import config

def setup_identify(model_path, n_iters=None, max_lang=None, thresh=None):
  global _identifier

  n_iters = n_iters if n_iters is not None else config.N_ITERS
  max_lang = max_lang if max_lang is not None else config.MAX_LANG
  thresh = thresh if thresh is not None else config.THRESHOLD
  _identifier = MultiLanguageIdentifier.from_modelpath(model_path, n_iters, max_lang, thresh)

def setup_default_identify(n_iters = None, max_lang=None, thresh=None):
  global _identifier

  n_iters = n_iters if n_iters is not None else config.N_ITERS
  max_lang = max_lang if max_lang is not None else config.MAX_LANG
  thresh = thresh if thresh is not None else config.THRESHOLD
  _identifier = MultiLanguageIdentifier.default(n_iters, max_lang, thresh)


def explain(path):
  """
  Explain the document as a distribution of tokens over the full language set.
  """
  global _identifier

  with open(path) as f:
    fv = _identifier.instance2fv(f.read())
    retval = _identifier.explain(fv)

  return [path] + list(retval)

def score(path):
  global _identifier

  with open(path) as f:
    p_score = _identifier.score(f.read())

  return [path] + list(p_score)

def identify(path):
  global _identifier

  with open(path) as f:
    text = f.read()
    try:
      pred = _identifier.identify(text)
    except ValueError:
      pred = []

  return [path] + pred

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--iters','-i',type=int, metavar='N', help="perform N iterations of Gibbs sampling", default=config.N_ITERS)
  parser.add_argument('--jobs','-j',type=int, metavar='N', help="use N processes", default=mp.cpu_count())
  parser.add_argument('--output','-o', help="output file (csv format)")
  parser.add_argument('--max_lang', type=int, help="maximum number of langugages to consider per-document", default=config.MAX_LANG)
  parser.add_argument('--thresh', '-t', type=float, help="threshold for including a language", default=config.THRESHOLD)
  parser.add_argument('--model', '-m', metavar="MODEL", help="path to model")

  group = parser.add_mutually_exclusive_group()
  group.add_argument('--explain', '-e', action='store_true', help="only explain documents as a breakdown over the full language set")
  group.add_argument('--dist', '-d', action='store_true', help="return distribution of values without thresholding")

  parser.add_argument('docs', metavar='FILE', help='file to process (read from stdin if blank)', nargs='*')

  args = parser.parse_args()

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

  if args.output:
    print >>sys.stderr, "processing {0} docs".format(len(doclist))


  if args.model:
    initalizer = setup_identify
    initargs = (args.model, args.iters, args.max_lang, args.thresh)
    langs = list(MultiLanguageIdentifier.list_langs(args.model))
  else:
    initalizer = setup_default_identify
    initargs = (args.iters, args.max_lang, args.thresh)
    langs = list(MultiLanguageIdentifier.list_langs())

  # Determine the type of output
  if args.dist:
    process = score
    writer.writerow(['path'] + langs)
  elif args.explain:
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
      if args.output:
        print >>sys.stderr, "** processed {0} docs in {1:.2f}s ({2:.2f} r/s) **".format(doc_count, t.elapsed, t.rate(doc_count) )

if __name__ == "__main__":
  main()
