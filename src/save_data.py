import cPickle
import os

data = "MovingPictures"
set = "1"

os.chdir("C:/Users/Derek/eclipse-workspace/NeuralNetwork/" + data + "/data_sets/" + set)

f = open(dir+"/training.save", 'wb')
cPickle.dump(training_data, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

f = open(dir+"/test.save", 'wb')
cPickle.dump(test_data, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

f = open(dir+"/validation.save", 'wb')
cPickle.dump(validation_data, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

