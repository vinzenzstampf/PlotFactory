! source /cvmfs/sft.cern.ch/lcg/views/LCG_94/x86_64-slc6-gcc8-opt/setup.sh 

from optparse import OptionParser
from DataCards import make_inputs as MI

parser = OptionParser()

parser.add_option("-ch", "--channel",
                  dest="channel",
                  type="string",
                  help="channel",
                  action="append",
                  default='mmm')

(options,args) = parser.parse_args()

print options, args
