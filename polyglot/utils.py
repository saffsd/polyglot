"""
Miscellaneous utilities.

Marco Lui <saffsd@gmail.com>, April 2013
"""

from contextlib import contextmanager, closing
import multiprocessing as mp
from itertools import imap

@contextmanager
def MapPool(processes=None, initializer=None, initargs=tuple(), maxtasksperchild=None, chunksize=1):
  """
  Contextmanager to express the common pattern of not using multiprocessing if
  only 1 job is allocated (for example for debugging reasons)
  """
  if processes is None:
    processes = mp.cpu_count() + 4

  if processes > 1:
    with closing( mp.Pool(processes, initializer, initargs, maxtasksperchild)) as pool:
      f = lambda fn, chunks: pool.imap_unordered(fn, chunks, chunksize=chunksize)
      yield f
  else:
    if initializer is not None:
      initializer(*initargs)
    f = imap
    yield f

  if processes > 1:
    pool.join()

from timeit import default_timer
class Timer(object):
  def __init__(self):
    self.timer = default_timer
    self.start = None
    self.end = None

  def __enter__(self):
    self.start = self.timer()
    self.end = None
    return self

  def __exit__(self, *args):
    self.end = self.timer()

  @property
  def elapsed(self):
    now = self.timer()
    if self.end is not None:
      self.end - self.start
    else:
      return now - self.start

  def rate(self, count):
    now = self.timer()
    if self.start is None:
      raise ValueError("Not yet started")

    return count / (now - self.start)

  def ETA(self, count, target):
    """
    Linearly estimate the ETA to reach target based on the current rate.
    """
    rate = self.rate(count)
    time_left = timedelta(seconds=int((target-count) / rate))
    return time_left 
