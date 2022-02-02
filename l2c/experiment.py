from identity import *
from quantize import *
from itcompress import *
from squish import *
from delta import *
from xor import *

ERROR_THRESH = 0.005

#set up baslines
BASELINES = []
BASELINES.append(Identity('none', error_thresh=ERROR_THRESH))
BASELINES.append(IdentityGZ('gz', error_thresh=ERROR_THRESH))
BASELINES.append(BitStripGZ('bs+gz', error_thresh=ERROR_THRESH))
BASELINES.append(Quantize('q', error_thresh=ERROR_THRESH))
BASELINES.append(QuantizeGZ('q+gz', error_thresh=ERROR_THRESH))
BASELINES.append(ItCompress('itcmp', error_thresh=ERROR_THRESH))
BASELINES.append(EntropyCoding('q+ent', error_thresh=ERROR_THRESH))
BASELINES.append(Sprintz('spz', error_thresh=ERROR_THRESH))
BASELINES.append(SprintzGzip('spz+gz', error_thresh=ERROR_THRESH))
BASELINES.append(Gorilla('grla', error_thresh=ERROR_THRESH))

data = np.loadtxt('/Users/sanjaykrishnan/Downloads/test_comp/ColorHistogram.asc')[:100,1:]

results = {}
for nn in BASELINES:
	nn.load(data)
	nn.compress()
	nn.decompress(data)

	results[nn.target] = nn.compression_stats



#plotting

import matplotlib.pyplot as plt

plt.bar([k for k in results], [results[k]['compressed_size'] for k in results])
plt.show()

