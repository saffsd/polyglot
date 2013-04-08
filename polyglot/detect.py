"""
Multi-langid based on a pre-trained P(w|t) and a Gibbs sampler for
estimating P(t|d).

Marco Lui, March 2013
"""
import argparse, sys, csv
import multiprocessing as mp

from identifier import MultiLanguageIdentifier
from utils import Timer, MapPool

def setup_identify(model_path, n_iters, max_lang, thresh):
  global __identifier
  __identifier = MultiLanguageIdentifier.from_modelpath(model_path, n_iters, max_lang, thresh)


def explain(path):
  global __identifier

  with open(path) as f:
    fv = __identifier.instance2fv(f.read())
    retval = __identifier.explain(fv)

  return [path] + list(retval)

def score(path):
  global __identifier

  with open(path) as f:
    p_score = __identifier.score(f.read())

  return [path] + list(p_score)

def identify(path):
  global __identifier

  with open(path) as f:
    text = f.read()
    try:
      pred = __identifier.identify(text)
    except ValueError:
      pred = []

  return [path] + pred

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--iters','-i',type=int, metavar='N', help="perform N iterations of Gibbs sampling", default=5)
  parser.add_argument('--jobs','-j',type=int, metavar='N', help="use N processes", default=mp.cpu_count())
  parser.add_argument('--output','-o', help="output file (csv format)")
  parser.add_argument('--max_lang', type=int, help="maximum number of langugages to consider per-document", default=10)
  parser.add_argument('--thresh', '-t', type=float, help="threshold for including a language", default=0.02)

  group = parser.add_mutually_exclusive_group()
  group.add_argument('--explain', '-e', action='store_true', help="only explain documents as a breakdown over the full language set")
  group.add_argument('--dist', '-d', action='store_true', help="return distribution of values without thresholding")

  parser.add_argument('model', metavar="MODEL", help="path to model")
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

  # Determine the type of output
  if args.dist:
    process = score
    writer.writerow(['path'] + list(MultiLanguageIdentifier.list_langs(args.model)))
  elif args.explain:
    process = explain 
    writer.writerow(['path'] + list(MultiLanguageIdentifier.list_langs(args.model)))
  else:
    process = identify


  # Process the documents specified
  doc_count = 0
  with MapPool(args.jobs, setup_identify, (args.model, args.iters, args.max_lang, args.thresh), chunksize=chunksize) as p, Timer() as t:
    for retval in p(process, doclist):
      writer.writerow( retval )
      doc_count += 1
      if args.output:
        print >>sys.stderr, "** processed {0} docs in {1:.2f}s ({2:.2f} r/s) **".format(doc_count, t.elapsed, t.rate(doc_count) )

if __name__ == "__main__":
  main()
