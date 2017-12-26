#!/usr/bin/python

from optparse import OptionParser
import sys
import os
import time
from sklearn.externals import joblib


if not os.environ.has_key('SONARHOME'):
    print 'Sonar environment not set'
    sys.exit(-1)
sonarhome = os.environ['SONARHOME']


parser = OptionParser()

parser.add_option("-b", "--bjob", dest="bjob",
                  help="which bjob to run", metavar="str")
parser.add_option("-n", "--novelty", dest="nov",
                  help="which novelty", metavar="str")
parser.add_option("-c", "--crossv", dest="crossv",
                  help="which cross validation file", metavar="str")
parser.add_option("-q", "--queue", dest="queue",
                  help="which bsub queue to use", metavar="str",default="2nw")
(opts, args) = parser.parse_args()
####################################################################
## Novelty
if opts.bjob is None:
    print 'Missing which bjob to run'
    sys.exit(-1)
if opts.nov is None:
    print 'Missing novelty class'
    sys.exit(-1)
if opts.crossv is None:
    cvfile = sonarhome + '/data/cvindexes_' + opts.nov + '_1024nfft.jbl'
else:
    cvfile = opts.crossv
if opts.queue is None:
    print 'Missing bsub queue'
    sys.exit(-1)
####################################################################
## GET NUMBER OF CROSS VALIDATION INDEXES
x = joblib.load(cvfile)
indexes = x['Indexes']
ncv = len(indexes)
####################################################################
## RUN A BSUB FOR EACH INDEXING
for icv in range(ncv):
    cmd = 'bsub -q %s bjob_schedule.sh %s %s %i'%(opts.queue, opts.bjob, opts.nov, icv)
    print 'Running: ', cmd
    fop = os.popen(cmd)
    fop_lines = fop.readlines()
    if len(fop_lines) == 0:
        print('Aborting')
        sys.exit(-1)
    fop.close()
print 'All jobs scheduled'



