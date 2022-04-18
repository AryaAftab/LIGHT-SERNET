import os

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib.font_manager as font_manager
from tqdm import tqdm

import hyperparameters



#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


	
class BestModelWeights(tf.keras.callbacks.Callback):
    def __init__(self, metric="val_accuracy", metric_type="max"):
        super(BestModelWeights, self).__init__()
        self.metric = metric
        self.metric_type = metric_type
        if self.metric_type not in ["min", "max"]:
        	  raise NameError('metric_type must be min or max')

    def on_train_begin(self, logs=None):
        if self.metric_type == "min":
            self.best_metric = np.inf
        else:
            self.best_metric = -np.inf
        self.best_epoch = 0
        self.model_best_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        if self.metric_type == "min":
            if self.best_metric >= logs[self.metric]:
                self.model_best_weights = self.model.get_weights()
                self.best_metric = logs[self.metric]
                self.best_epoch = epoch
        else:
          if self.best_metric <= logs[self.metric]:
              self.model_best_weights = self.model.get_weights()
              self.best_metric = logs[self.metric]
              self.best_epoch = epoch

    def on_train_end(self, logs=None):
        self.model.set_weights(self.model_best_weights)
        print(f"\nBest weights is set, Best Epoch was : {self.best_epoch+1}\n")
	
	
	
class ShowProgress(tf.keras.callbacks.Callback):
    def __init__(self, epochs, step_show=1, metric="accuracy"):
        super(ShowProgress, self).__init__()
        self.epochs = epochs
        self.step_show = step_show
        self.metric = metric

    def on_train_begin(self, logs=None):
        self.pbar = tqdm(range(self.epochs))

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.step_show == 0:
            self.pbar.set_description(f"Epoch : {epoch + 1} / {self.epochs}, Train {self.metric} : {round(logs[self.metric], 4)}, Valid {self.metric} : {round(logs['val_' + self.metric], 4)}")
            self.pbar.update(self.step_show)



def LearningRateScheduler():
	def scheduler(epoch, lr):
	    if epoch < hyperparameters.LEARNING_RATE_DECAY_STRATPOINT:
	        return lr
	    else:
	        if epoch % hyperparameters.LEARNING_RATE_DECAY_STEP == 0:
	            lr = lr * tf.math.exp(hyperparameters.LEARNING_RATE_DECAY_PARAMETERS)
	    return lr
	return tf.keras.callbacks.LearningRateScheduler(scheduler)



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
	"""
	given a sklearn confusion matrix (cm), make a nice plot

	Arguments
	---------
	cm:           confusion matrix from sklearn.metrics.confusion_matrix

	target_names: given classification classes such as [0, 1, 2]
	              the class names, for example: ['high', 'medium', 'low']

	title:        the text to display at the top of the matrix

	cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
	              see http://matplotlib.org/examples/color/colormaps_reference.html
	              plt.get_cmap('jet') or plt.cm.Blues

	normalize:    If False, plot the raw numbers
	              If True, plot the proportions

	Usage
	-----
	plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
	                                                          # sklearn.metrics.confusion_matrix
	                      normalize    = True,                # show proportions
	                      target_names = y_labels_vals,       # list of names of the classes
	                      title        = best_estimator_name) # title of graph

	Citiation
	---------
	http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

	"""


	legend_font = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=15)
	CM_axis_font = {'fontname':'Times New Roman', 'size':'14', 'weight':'bold', 'color':'black'}
	CM_tick_font = {'fontname':'Times New Roman', 'size':'10', 'weight':'bold', 'color':'black'}
	AL_axis_font = {'fontname':'Times New Roman', 'size':'18', 'weight':'bold', 'color':'black'}
	AL_tick_font = {'fontname':'Times New Roman', 'size':'13', 'weight':'bold', 'color':'black'}


	accuracy = np.trace(cm) / float(np.sum(cm))
	misclass = 1 - accuracy

	if cmap is None:
		cmap = plt.get_cmap('Blues')

	#plt.figure(figsize=(8, 6))
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	#plt.title(title)
	plt.colorbar()

	if target_names is not None:
		tick_marks = np.arange(len(target_names))
		plt.xticks(tick_marks, target_names, rotation=45)
		plt.yticks(tick_marks, target_names)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


	thresh = cm.max() / 1.5 if normalize else cm.max() / 2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		if normalize:
			plt.text(j, i, "{:0.4f}".format(cm[i, j]),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black")
		else:
			plt.text(j, i, "{:,}".format(cm[i, j]),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black")


	plt.tight_layout()
	plt.ylabel('True label', **CM_axis_font)
	plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), **CM_axis_font)
	plt.xticks(rotation=90, **CM_tick_font) 
	plt.yticks(**CM_tick_font)