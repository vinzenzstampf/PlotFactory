from modules.fr_net import make_all_friendtrees, path_to_NeuralNet
from socket import gethostname
from pdb import set_trace

hostname = gethostname()
faketype = 'nonprompt'

if "lxplus" in hostname:
    analysis_dir = '/eos/user/v/vstampf/ntuples/'
if "t3ui" in hostname:
    analysis_dir = '/work/dezhu/4_production/'
if "starseeker" in hostname:
    analysis_dir = '/home/dehuazhu/SESSD/4_production/'

## select here the channel you want to analyze
channel = 'mmm'    
# channel = 'eee'    
# channel = 'eem'
# channel = 'mem'

path_to_NeuralNet = path_to_NeuralNet(faketype, channel) 

make_all_friendtrees(
        multiprocess = False,
        server = hostname,
        analysis_dir = analysis_dir,
        channel=channel,
        path_to_NeuralNet = path_to_NeuralNet,
        overwrite = False,
        )
