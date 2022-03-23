from identity import *
from quantize import *
from itcompress import *
from squish import *
from delta import *
from xor import *
from spartan import *
from apca import *
from linear import *

import numpy as np

def initialize(ERROR_THRESH = 1e-4):
	#set up baslines
	BASELINES = []
	BASELINES.append(IdentityGZ('gz', error_thresh=ERROR_THRESH))
	BASELINES.append(Quantize('q', error_thresh=ERROR_THRESH))
	BASELINES.append(QuantizeGZ('q+gz', error_thresh=ERROR_THRESH))
	#BASELINES.append(ItCompress('itcmp', error_thresh=ERROR_THRESH))
	#BASELINES.append(Spartan('sptn', error_thresh=ERROR_THRESH))
	BASELINES.append(EntropyCoding('q+ent', error_thresh=ERROR_THRESH))
	#BASELINES.append(Sprintz('spz', error_thresh=ERROR_THRESH))
	#BASELINES.append(SprintzGzip('spz+gz', error_thresh=ERROR_THRESH))
	#BASELINES.append(Gorilla('grla', error_thresh=ERROR_THRESH))
	#BASELINES.append(GorillaLossy('grla+l', error_thresh=ERROR_THRESH))
	BASELINES.append(AdaptivePiecewiseConstant('apca', error_thresh=ERROR_THRESH))
	BASELINES.append(MultivariateHierarchical('hier', error_thresh=ERROR_THRESH, blocksize=4096))
	return BASELINES

def run(BASELINES,\
		DATA_DIRECTORY = '/Users/sanjaykrishnan/Downloads/HT_Sensor_UCIsubmission/', \
		FILENAME = 'HT_Sensor_dataset.dat',\
		N=4096):
	#orig = np.loadtxt(DATA_DIRECTORY + FILENAME)[:N,1:]
	orig = np.load('/Users/sanjaykrishnan/Downloads/l2c/data/exchange_rate.npy')[:N,1:]

	#print(orig[:100,0])
	#exit()

	bresults = {}
	for nn in BASELINES:
		data = orig.copy()
		nn.load(data)
		nn.compress()
		nn.decompress(data)
		bresults[nn.target] = nn.compression_stats
		print(nn.target, nn.error_thresh, nn.compression_stats['errors'])

	return bresults





#plotting

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = (10,4)


BASELINES = initialize()
FILENAME = 'exchange_rate'
#FILENAME = 'ht'
SIZE_LIMIT = 4096
bresults = run(BASELINES, N=SIZE_LIMIT)


#compressed size
plt.figure()
plt.title(FILENAME.split('.')[0] + ": Compression Ratio" )
plt.ylabel('Compression Ratio')
plt.bar([k for k in bresults], [bresults[k]['compressed_ratio'] for k in bresults])
plt.bar([k for k in bresults], [bresults[k].get('model_size',0)/bresults[k].get('original_size') for k in bresults])
plt.legend(['Compression Ratio', 'Model Contribution'])
plt.savefig('compression_ratio_' + FILENAME.split('.')[0]+'.png')



#compression throughput (subtract bitpacking time)
plt.figure()
plt.title(FILENAME.split('.')[0] + ": Throughput" )
plt.ylabel('Thpt bytes/sec')

x1 = [i - 0.1 for i,_ in enumerate(bresults)]
x2 = [i + 0.1 for i,_ in enumerate(bresults)]
x = [i for i,_ in enumerate(bresults)]
labels = [k for _,k in enumerate(bresults)]

plt.bar(x1, [bresults[k]['original_size']/(bresults[k]['compression_latency'])  for k in bresults], width=0.2)
plt.bar(x2, [bresults[k]['compressed_size']/(bresults[k]['decompression_latency'])  for k in bresults], width=0.2)

plt.xticks(x,labels)
plt.legend(['Compression', 'Decompression'])
plt.yscale('log')
plt.yticks([10000,100000,1000000, 10000000,1e8])
plt.savefig('compression_tpt_' + FILENAME.split('.')[0]+'.png')


#compression curves
plt.figure()
results = {}
for error_thresh in range(7,0,-1):
	BASELINES = initialize(ERROR_THRESH=10**(-error_thresh))
	output = run(BASELINES, N=SIZE_LIMIT)
	for k in output:

		if not k in results:
			results[k] = [None]*7

		results[k][ (7-error_thresh) ] = output[k]


ax = plt.axes()
ax.set_prop_cycle('color',[plt.cm.tab20(i) for i in np.linspace(0, 1, len(results))])

for technique in results:
	rgb = np.random.rand(3,)
	plt.plot([v['compressed_ratio'] for v in results[technique]],'s-')

plt.legend([technique for technique in results])
plt.xticks(list(range(0,7)),[ "1e-"+str(r) for r in range(7,0,-1)])
plt.xlim([0,7])
plt.xlabel('Error Threshold %')
plt.title(FILENAME.split('.')[0] + ": Error Dependence" )
plt.ylabel('Compression Ratio')
plt.savefig('thresh_v_ratio' + FILENAME.split('.')[0]+'.png')



