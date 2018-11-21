import tensorflow as tf, sys,os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import sys

image_dir = sys.argv[2]

female_image_dir = image_dir+'/female'

male_image_dir = image_dir+'/male'

female_image_paths = [x for x in filter(lambda filename:filename.endswith('jpeg'),os.listdir(female_image_dir))]

female_image_datas = {}

for female_image_path in female_image_paths:
	female_image_datas[female_image_path.split('.jpeg')[0]] = tf.gfile.FastGFile(female_image_dir+'/'+female_image_path,'rb').read()
	
male_image_paths = [x for x in filter(lambda filename:filename.endswith('jpeg'),os.listdir(male_image_dir))]

male_image_datas = {}

for male_image_path in male_image_paths:
	male_image_datas[male_image_path.split('.jpeg')[0]] = tf.gfile.FastGFile(male_image_dir+'/'+male_image_path,'rb').read()

root_path = sys.argv[1]

label_lines = [line.rstrip() for line in tf.gfile.GFile(root_path+'retrain_labels.txt')]


with tf.gfile.FastGFile(root_path+'retrain_graph.pb','rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	_ = tf.import_graph_def(graph_def,name="")

prediction_arr = []
actual_arr = []
with tf.Session() as sess:
	softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
	real_male = len(male_image_datas)
	real_female = len(female_image_datas)
	p_male = 0
	p_female = 0
	count1 = 0
	threshold = 0.5

	for key in female_image_datas:
		predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0':female_image_datas[key]})
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
		score = predictions[0][0]
		prediction_arr.append(score)
		actual_arr.append(0)

		if score > threshold:
			result = 'male'
			p_male+=1
		else:
			result = 'female'
			score = 1-score
			count1+=1
			p_female+=1

		print('%s(score = %.5f)'%(result,score))

	count2 = 0
	print()

	for key in male_image_datas:
		predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0':male_image_datas[key]})
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
		score = predictions[0][0]
		prediction_arr.append(score)
		actual_arr.append(1)

		if score > threshold:
			result = 'male'
			count2+=1
			p_male+=1
		else:
			result = 'female'
			score = 1-score
			p_female+=1
			

		print('%s(score = %.5f)'%(result,score))
	print('Male Recall:{0}'.format(count2/real_male))
	print('Female Recall:{0}'.format(count1/real_female))
	print('Female Precision:%.5f'%(count1/p_female))
	print('Male Precision:%.5f'%(count2/p_male))
	print('Overall ACC:%.5f'%((count1+count2)/(real_male+real_female)))
	print('Male successful predicted:{0}'.format(count2))
	print('female successful predicted:{0}'.format(count1))
	print('Actucal male{0}'.format(real_male))
	print('Actucal female{0}'.format(real_female))


false_positive_rate, true_positive_rate, thresholds = roc_curve(actual_arr, prediction_arr)
roc_auc = auc(false_positive_rate, true_positive_rate)
fig,ax = plt.subplots()
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()	
fig.savefig('tensorflow_testset_roc.eps',format='eps',dpi=1200)