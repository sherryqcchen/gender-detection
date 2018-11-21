import sys, os
sys.path.append("../..")
# import facerec modules
from facerec.feature import Fisherfaces
from facerec.distance import EuclideanDistance
from facerec.classifier import NearestNeighbor
from facerec.classifier import SVM
from facerec.svm import grid_search as gs
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.validation import LeaveOneOutCrossValidation
from facerec.visual import subplot
from facerec.util import minmax_normalize
# import numpy, matplotlib and logging
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import logging

def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given)
                    if (sz is not None):
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError as e:
                    print ("I/O error({0}):".format(e))
                except:
                    print ("Unexpected error:", sys.exc_info()[0])
                    raise
            c = c+1
    return [X,y]

# if __name__ == "__main__":
def run():
    # This is where we write the images, if an output_dir is given
    # in command line:

    # out_dir = None

    # You'll need at least a path to your image data, please see
    # the tutorial coming with this source code on how to prepare
    # your image data:

    # if len(sys.argv) < 2:
    #     print ("USAGE: facerec_demo.py </path/to/images>")
    #     sys.exit()

    # Now read in the image data. This must be a valid path!

    # [X,y] = read_images(sys.argv[1])
    [X,y] = read_images('../data/trainset/')


    # dataset = FilesystemReader(sys.argv[1])
    # Then set up a handler for logging:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add handler to facerec modules, so we see what's going on inside:
    logger = logging.getLogger("facerec")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # Define the Fisherfaces as Feature Extraction method:
    feature = Fisherfaces()
    # Define a 1-NN classifier with Euclidean Distance:
    svm = SVM(C=0.1, kernel='rbf', degree=4, gamma='auto', coef0=0.0)
    knn = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)
    # # Define the model as the combination
    model_svm = PredictableModel(feature=feature, classifier=svm)

    model_knn = PredictableModel(feature=feature,classifier=knn)

    # # Compute the Fisherfaces on the given data (in X) and labels (in y):
    model_svm.compute(X, y)

    model_knn.compute(X, y)
    # E = []
    # for i in range(min(model.feature.eigenvectors.shape[1], 16)):
	   #  e = model.feature.eigenvectors[:,i].reshape(X[0].shape)
	   #  E.append(minmax_normalize(e,0,255, dtype=np.uint8))
    # subplot(title="Fisherfaces", images=E, rows=4, cols=4, sptitle="Fisherface", colormap=cm.jet, filename="fisherfaces.png")

    # cv = LeaveOneOutCrossValidation(model)
    # print(cv0)
    # cv0.validate(dataset.data,dataset.classes,print_debug=True)
    cv_svm = KFoldCrossValidation(model_svm, k=10)
    cv_knn = KFoldCrossValidation(model_knn, k=10)

    param_grid = [
      {'C': [0.05, 0.1, 0.3, 0.5,1,2,5], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
     ]
    [tX,tY] = read_images('../data/testset/')

    # cv_svm.validate(X, y)
    # cv_knn.validate(X, y)

    gs(model_svm,X,y,param_grid)

    count1 = 0
    count2 = 0
    for i in range(len(tY)):
        r1 = model_svm.predict(tX[i])
        r2 = model_knn.predict(tX[i])
        if r1[0] == tY[i]:
            count1+=1
        if r2[0] == tY[i]:
            count2+=1

    print('SVM ACC:{0}'.format(count1/len(tY)))
    print('KNN ACC:{0}'.format(count2/len(tY)))
    print(cv_knn.print_results())
    print(cv_svm.print_results())

run()
