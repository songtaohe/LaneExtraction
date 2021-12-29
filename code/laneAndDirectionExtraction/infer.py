import sys 

from model import LaneModel

from PIL import Image 
import numpy as np 
from subprocess import Popen 
import tensorflow as tf 
import scipy.ndimage

inputfile = sys.argv[1]
outputfolder = sys.argv[2]
backbone = sys.argv[3]
windowsize1 = 256
windowsize2 = 512
cnninput = 640

margin = (cnninput - windowsize1) // 2

margin2 = (cnninput - windowsize2) // 2

Popen("mkdir -p " + outputfolder, shell=True).wait()

img = scipy.ndimage.imread(inputfile)
sdmap = scipy.ndimage.imread(inputfile.replace("sat", "sdmap"))

img = (img.astype(np.float) / 255.0 - 0.5) * 0.81 
sdmap = sdmap.astype(np.float) / 255.0
dim = np.shape(img)

img = np.pad(img, ((margin, margin),(margin, margin),(0,0)), 'constant')
sdmap = np.pad(sdmap, ((margin, margin),(margin, margin)), 'constant')

mask = np.zeros((cnninput,cnninput,3)) 
for i in range((windowsize2 - windowsize1) // 2 ):
	r = i / float((windowsize2 - windowsize1) // 2)
	mask[margin2+i:-(margin2+i-1),margin2+i:-(margin2+i-1),:] = r 


output = np.zeros_like(img)
weights = np.zeros_like(img) + 0.0001

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	model = LaneModel(sess, cnninput, batchsize=1, sdmap=False, backbone = backbone)
	#model.restoreModel("modelrun3_640_resnet34v3/model196900")
	#model.restoreModel("modelfinetune_run1_640_resnet34v3/model52215")
	#model.restoreModel("modelfinetune_run2_640_resnet34v3/model52215")
	


	x_in = np.zeros((1, cnninput, cnninput, 3))
	x_in2 = np.zeros((1, cnninput, cnninput, 1))


	for model_ep in [490]:
		if backbone == "resnet34v3":
			model.restoreModel("model_4cities_run2_640_%s_500ep/model%d" % (backbone, model_ep))
		else:
			model.restoreModel("model_4cities_run1_640_%s_500ep/model%d" % (backbone, model_ep))

		for i in range(dim[0] // windowsize1):
			print(i)
			for j in range(dim[1] // windowsize1):
				
				r = i * windowsize1
				c = j * windowsize1
				x_in[0,:,:,:] = img[r:r+cnninput, c:c+cnninput,:] 
				x_in2[0,:,:,0] = sdmap[r:r+cnninput, c:c+cnninput]

				x_out = model.infer(x_in)[0]

				output[r:r+cnninput, c:c+cnninput,:] += x_out[0,:,:,:] * mask
				weights[r:r+cnninput, c:c+cnninput,:] += mask[:,:,0:1]

	
	output = np.divide(output, weights)

	output = output[margin:-margin, margin:-margin,:]
	Image.fromarray(((output[:,:,0]) * 255).astype(np.uint8) ).save(outputfolder + "/seg.png")

	direction_img = np.zeros(dim, dtype=np.uint8)

	direction_img[:,:,2] = np.clip(output[:,:,1],-1,1) * 127 + 127
	direction_img[:,:,1] = np.clip(output[:,:,2],-1,1) * 127 + 127
	direction_img[:,:,0] = 127
	
	Image.fromarray(direction_img.astype(np.uint8) ).save(outputfolder + "/direction.png")
	


