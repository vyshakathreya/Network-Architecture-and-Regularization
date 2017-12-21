
import numpy as np

from keras.callbacks import Callback
from keras.layers import Dense, Input
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras import metrics

from sklearn.model_selection import StratifiedKFold

from dsp.utils import Timer

class ErrorHistory(Callback):
    def on_train_begin(self, logs={}):
        self.error = []

    def on_batch_end(self, batch, logs={}):
        self.error.append(100 - logs.get('acc'))


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def feed_forward_model(specification):
    """feed_forward_model - specification list
    Create a feed forward model given a specification list
    Each element of the list represents a layer and is formed by a tuple.
    
    (layer_constructor, 
     positional_parameter_list,
     keyword_parameter_dictionary)
    
    Example, create M dimensional input to a 3 layer network with 
    20 unit ReLU hidden layers and N unit softmax output layer
    
    [(Dense, [20], {'activation':'relu', 'input_dim': M}),
     (Dense, [20], {'activation':'relu', 'input_dim':20}),
     (Dense, [N], {'activation':'softmax', 'input_dim':20})
    ]

    """
    model = Sequential()
    
    for item in specification:
        layertype = item[0]
        # Construct layer and add to model
        # This uses Python's *args and **kwargs constructs
        #
        # In a function call, *args passes each item of a list to 
        # the function as a positional parameter
        #
        # **args passes each item of a dictionary as a keyword argument
        # use the dictionary key as the argument name and the dictionary
        # value as the parameter value
        #
        # Note that *args and **args can be used in function declarations
        # to accept variable length arguments.
        layer = layertype(*item[1], **item[2])
        model.add(layer)
        #model.add(Dense(10, activation='relu', input_dim = 30))
    return model
        
class CrossValidator:
    debug = False
    
    def __init__(self, Examples, Labels, model_spec, n_folds=10, epochs=100): 
        """CrossValidator(Examples, Labels, model_spec, n_folds, epochs)
        Given a list of training examples in Examples and a corresponding
        set of class labels in Labels, train and evaluate a learner
        using cross validation.
        
        arguments:
        Examples:  feature matrix, each row is a feature vector
        Labels:  Class labels, one per feature vector
        n_folds:  Number of folds in experiment
        epochs:  Number of times through data set
        model_spec: Specification of model to learn, see 
            feed_forward_model() for details and example  
        
        """
        
        # Create a plan for k-fold testing with shuffling of examples
        # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html    #
        kfold = StratifiedKFold(n_folds, shuffle=True)
        
    
        foldidx = 0
        errors  = np.zeros([n_folds, 1])
        models = []
        losses = []
        timer = Timer()
        for (train_idx, test_idx) in kfold.split(Examples, Labels):
            (errors[foldidx], model, loss) = self.train_and_evaluate__model(
                Examples, Labels, train_idx, test_idx, model_spec) 
            models.append(model)
            losses.append(loss)
# =============================================================================
#             print(
#                 "Fold {} error {}, cumulative cross-validation time {}".format(
#                     foldidx, errors[foldidx], timer.elapsed()))
# =============================================================================
            foldidx = foldidx + 1
        
        # Show architecture of last model (all are the same)    
        print("Model summary\n{}".format(model.summary()))
        
        print("Fold errors:  {}".format(errors))
        print("Mean error {} +- {}".format(np.mean(errors), np.std(errors)))
        
        print("Experiment time: {}".format(timer.elapsed()))
        
        self.errors = errors
        self.models = models
        self.losses = losses

    def train_and_evaluate__model(self, examples, labels, train_idx, test_idx, 
                                  model_spec, batch_size=100, epochs=100):
        """train_and_evaluate__model(examples, labels, train_idx, test_idx,
                model_spec, batch_size, epochs)
                
        Given:
            examples - List of examples in column major order
                (# of rows is feature dim)
            labels - list of corresponding labels
            train_idx - list of indices of examples and labels to be learned
            test_idx - list of indices of examples and labels of which
                the system should be tested.
            model_spec - Model specification, see feed_forward_model
                for details and example
        Optional arguments
            batch_size - size of minibatch
            epochs - # of epochs to compute
            
        Returns error rate, model, and loss history over training
        """
    
        # Convert labels to a one-hot vector
        # https://keras.io/utils/#to_categorical
        onehotlabels = np_utils.to_categorical(labels)

        # Get dimension of model
        dim = examples.shape[1]

        error = ErrorHistory()
        loss = LossHistory()
        

        model = feed_forward_model(model_spec)
        
        model.compile(optimizer = "Adam", 
                      loss = "categorical_crossentropy",
                      metrics = [metrics.categorical_accuracy])
        
        if CrossValidator.debug:
            model.summary()  # display
        
        # Train the model
        model.fit(examples[train_idx], onehotlabels[train_idx], 
                  batch_size=batch_size, epochs=epochs,
                  callbacks = [loss], verbose = CrossValidator.debug)
        #print("Training loss %s"%(["%f"%(loss) for loss in loss.losses]))
        
        
        result = model.evaluate(examples[test_idx], onehotlabels[test_idx], 
                                 verbose=CrossValidator.debug)

        return (1 - result[1], model, loss) 
        
      
    def get_models(self):
        "get_models() - Return list of models created by cross validation"
        return self.models
    
    def get_errors(self):
        "get_errors - Return list of error rates from each fold"
        return self.errors
    
    def get_losses(self):
        "get_losses - Return list of loss histories associated with each model"
        return self.losses
          
        
        


    