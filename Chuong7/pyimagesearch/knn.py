from datasets.simpledatasetloader import SimpleDatasetLoader
from preprocessing.simplepreprocessor import Simplepreprocessor

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imutils import paths
import argparse

ap= argparse.ArgumentParser()

ap.add_argument("-d","--dataset",help="animals/",default="Chuong7/pyimagesearch/animals")
ap.add_argument("-k","--neighbors",type=int, default=1,help="The number of neighbors for classification ")
ap.add_argument("-j","--jobs",type=int, default=-1, help="The number of jobs for K-NN distance")
args = vars(ap.parse_args())

print("Loading Images... ")
imagePaths= list(paths.list_images(args["dataset"]))

sp= Simplepreprocessor(32,32)
sdl= SimpleDatasetLoader(preprocessors=[sp])
(data,labels) = sdl.load(imagePaths,verbose=500)

data = data.reshape((data.shape[0],32*32*3))

print("Feature Matrix :{:.1f}MB".format(data.nbytes/(1024*1000.0)))

le= LabelEncoder()
labels=le.fit_transform(labels)

(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.25, random_state=42)

print("[INFO] evaluating k-NN classifier...")
model=KNeighborsClassifier(n_neighbors=args["neighbors"],n_jobs=args["jobs"])

model.fit(trainX,trainY)
y_pred=model.predict(testX)
print(classification_report(testY,y_pred,target_names=le.classes_))

print("acc:",accuracy_score(testY,y_pred))