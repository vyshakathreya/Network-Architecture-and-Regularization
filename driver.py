from dsp.pca import PCA
from dsp.utils import pca_analysis_of_spectra
from dsp.utils import get_corpus, get_class, Timer, \
    extract_features_from_corpus
from dsp.features import get_features
from classifier.feedforward import CrossValidator

from keras.layers import Dense, Dropout
from keras.utils import np_utils
import numpy as np
import keras 
import matplotlib.pyplot as mp      
   
def main():
   
    
    files = get_corpus("C:/Users/vysha/Downloads/wav/train")
    # for testing
    if False:
        files[50:] = []  # truncate test for speed
    
    print("%d files"%(len(files)))
    
    adv_ms = 10
    len_ms = 20
    # We want to retain offset_s about the center
    offset_s = 0.25    

    timer = Timer()
    pca = pca_analysis_of_spectra(files, adv_ms, len_ms, offset_s)
    print("PCA feature generation and analysis time {}, feature extraction..."
          .format(timer.elapsed()))
    
    timer.reset()
    # Read features - each row is a feature vector
    components = 40
    examples = extract_features_from_corpus(
        files, adv_ms, len_ms, offset_s, pca, components)        
    print("Time to generate features {}".format(timer.elapsed()))
    timer.reset()
    
    labels = get_class(files)
    outputN = len(set(labels))
    
    # Specify model architectures
    models = [       
        # 3 layer 20x20xoutput baseline (problem set 3)
        [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1]}),
         (Dense, [20], {'activation':'relu', 'input_dim':20}),
         (Dense, [outputN], {'activation':'softmax', 'input_dim':20})
        ],
        [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1]}),
         (Dense, [40], {'activation':'relu', 'input_dim':20}),
         (Dense, [outputN], {'activation':'softmax', 'input_dim':40})
        ],
        [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1]}),
         (Dense, [20], {'activation':'relu', 'input_dim':20}),
         (Dense, [40], {'activation':'relu', 'input_dim':20}),
         (Dense, [outputN], {'activation':'softmax', 'input_dim':40})
        ],
        [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1]}),
         (Dropout, [0.3], {}),
         (Dense, [20], {'activation':'relu', 'input_dim':20}),
         (Dropout, [0.3], {}),
         (Dense, [40], {'activation':'relu', 'input_dim':20}),
         (Dense, [outputN], {'activation':'softmax', 'input_dim':40})
        ],
        [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1]}),
         (Dropout, [0.2], {}),
         (Dense, [20], {'activation':'relu', 'input_dim':20}),
         (Dropout, [0.2], {}),
         (Dense, [40], {'activation':'relu', 'input_dim':20}),
         (Dense, [outputN], {'activation':'softmax', 'input_dim':40})
        ],
        [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1]}),
         (Dense, [20], {'activation':'relu', 'input_dim':20,'kernel_regularizer': keras.regularizers.l1(0.01)}),
         (Dense, [40], {'activation':'relu', 'input_dim':20, 'kernel_regularizer': keras.regularizers.l1(0.01)} ),
         (Dense, [outputN], {'activation':'softmax', 'input_dim':40})
        ],
        [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1]}),
         (Dense, [20], {'activation':'relu', 'input_dim':20,'kernel_regularizer': keras.regularizers.l2(0.01)}),
         (Dense, [40], {'activation':'relu', 'input_dim':20, 'kernel_regularizer': keras.regularizers.l2(0.01)}),
         (Dense, [outputN], {'activation':'softmax', 'input_dim':40})
        ],
        [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1]}),
         (Dense, [20], {'activation':'relu', 'input_dim':20,'kernel_regularizer': keras.regularizers.l1(0.01)}),
         (Dense, [40], {'activation':'relu', 'input_dim':20, 'kernel_regularizer': keras.regularizers.l1(0.01)}),
         (Dense, [outputN], {'activation':'softmax', 'input_dim':40})
        ],
        [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1]}),
         (Dense, [20], {'activation':'relu', 'input_dim':10,'kernel_regularizer': keras.regularizers.l1(0.01)}),
         (Dense, [outputN], {'activation':'softmax', 'input_dim':20})
        ],
        [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1]}),
         (Dense, [20], {'activation':'relu', 'input_dim':10,'kernel_regularizer': keras.regularizers.l1(0.01)}),
         (Dense, [outputN], {'activation':'softmax', 'input_dim':20})
        ],
        [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1]}),
         (Dense, [20], {'activation':'relu', 'input_dim':10,'kernel_regularizer': keras.regularizers.l2(0.01)}),
         (Dense, [outputN], {'activation':'softmax', 'input_dim':20,})
        ],
        [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1]}),
         (Dense, [20], {'activation':'relu', 'input_dim':10,'kernel_regularizer': keras.regularizers.l2(0.01)}),
         (Dense, [outputN], {'activation':'softmax', 'input_dim':20})
        ],
        [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1]}),
          (Dense, [20], {'activation':'relu', 'input_dim':20}),
          (Dense, [outputN], {'activation':'softmax', 'input_dim':20})
         ],
         [(Dense, [10], {'activation':'relu', 'input_dim':examples.shape[1]}),
          (Dense, [10], {'activation':'relu', 'input_dim':10}),
          (Dense, [10], {'activation':'relu', 'input_dim':10}),
          (Dense, [10], {'activation':'relu', 'input_dim':10}),
          (Dense, [outputN], {'activation':'softmax', 'input_dim':10})
         ],
         [(Dense, [5], {'activation':'relu', 'input_dim':examples.shape[1]}),
          (Dense, [5], {'activation':'relu', 'input_dim':5}),
          (Dense, [5], {'activation':'relu', 'input_dim':5}),
          (Dense, [5], {'activation':'relu', 'input_dim':5}),
          (Dense, [10], {'activation':'relu', 'input_dim':5}),
          (Dense, [10], {'activation':'relu', 'input_dim':10}),
          (Dense, [outputN], {'activation':'softmax', 'input_dim':10})
         ]
        # Add more models here...  [(...), (...), ...], [(...), ...], ....
        ]
    
    print("Time to build matrix {}, starting cross validation".format(
        timer.elapsed()))
    
# =============================================================================
#     compare L1 and L2
# =============================================================================
# =============================================================================
#     models.clear()
#     models = [
#         [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1]}),
#          (Dense, [20], {'activation':'relu', 'input_dim':20}),
#          (Dense, [outputN], {'activation':'softmax', 'input_dim':20})
#         ],
#         [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1], 
#           'kernel_regularizer': keras.regularizers.l1(0.01)}),
#          (Dense, [20], {'activation':'relu', 'input_dim':10,'kernel_regularizer': keras.regularizers.l1(0.01)}),
#          (Dense, [outputN], {'activation':'softmax', 'input_dim':20})
#         ],
#         [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1], 
#           'kernel_regularizer': keras.regularizers.l2(0.01)}),
#          (Dense, [20], {'activation':'relu', 'input_dim':10,'kernel_regularizer': keras.regularizers.l2(0.01)}),
#          (Dense, [outputN], {'activation':'softmax', 'input_dim':20,})
#         ]
#         ]
# =============================================================================
    
    debug = False
    c = []
    if debug: 
        c.append(CrossValidator(examples, labels, models[2], epochs=50))
    else:
        for architecture in models:
            c.append(CrossValidator(examples, labels, architecture, epochs=100))
    # do something useful with c... e.g. generate tables/graphs, etc.
    avg_Err = []
    std_Err = []
    count = 0
    
    for a in c:
        avg_Err.append(np.average(a.get_errors()))
        count = count + 1
    
    layers = np.arange(count)
    print(avg_Err)
    
# =============================================================================
#     files_test = get_corpus("C:/Users/vysha/Downloads/wav/test")
#     print("%d test files"%(len(files_test)))
#     
#     examples_test = extract_features_from_corpus(
#         files_test, adv_ms, len_ms, offset_s, pca, components)        
#     
#     print("Time to generate test features {}".format(timer.elapsed()))
#     timer.reset()
#     
#     labels_test = np_utils.to_categorical(get_class(files_test))
#     
#     accu = []
#     avgAccu = []
#     for m in c:
#         model = m.models
#         for n in model:
#             y = n.evaluate(np.array(examples_test),np.array(labels_test),verbose = 0)
#             accu.append(1-y[1])
#         avgAccu.append(np.average(accu))
# =============================================================================
    
    
    mp.figure()
#    p1 = mp.bar(layers,avgAccu)
    p2 = mp.bar(layers,avg_Err)
    mp.title('L1 and L2 analysis')
    mp.xlabel('Expt number')
    mp.ylabel('Error')
 #   mp.xticks(layers, ('Unregularized', 'L1', 'L2'))
#    mp.legend((p1[0], p2[0]), ('Test', 'Train'))
    
    
# =============================================================================
#     "Study depth vs error" 
# =============================================================================
    
# =============================================================================
#     models = [
#          [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1]}),
#           (Dense, [20], {'activation':'relu', 'input_dim':20}),
#           (Dense, [outputN], {'activation':'softmax', 'input_dim':20})
#          ],
#          [(Dense, [10], {'activation':'relu', 'input_dim':examples.shape[1]}),
#           (Dense, [10], {'activation':'relu', 'input_dim':10}),
#           (Dense, [10], {'activation':'relu', 'input_dim':10}),
#           (Dense, [10], {'activation':'relu', 'input_dim':10}),
#           (Dense, [outputN], {'activation':'softmax', 'input_dim':10})
#          ],
#          [(Dense, [5], {'activation':'relu', 'input_dim':examples.shape[1]}),
#           (Dense, [5], {'activation':'relu', 'input_dim':5}),
#           (Dense, [5], {'activation':'relu', 'input_dim':5}),
#           (Dense, [5], {'activation':'relu', 'input_dim':5}),
#           (Dense, [10], {'activation':'relu', 'input_dim':5}),
#           (Dense, [10], {'activation':'relu', 'input_dim':10}),
#           (Dense, [outputN], {'activation':'softmax', 'input_dim':10})
#          ],        
#          ]
#          
#     c=[]
#     for architecture in models:
#         c.append(CrossValidator(examples, labels, architecture, epochs=100))
#         
#     
#     # do something useful with c... e.g. generate tables/graphs, etc.
#     avg_Err = []
#     std_Err = []
#     count = 0
#     
#     for a in c:
#         avg_Err.append(np.average(a.get_errors()))
#         std_Err.append(np.std(a.get_errors()))
#         count = count + 1
#     
#     layers = [2,4,6]
#     print(avg_Err)
#     
#     accu = []
#     avgAccu = []
#     for m in c:
#         model = m.models
#         for n in model:
#             y = n.evaluate(np.array(examples_test),np.array(labels_test),verbose = 0)
#             accu.append(1-y[1])
#         avgAccu.append(np.average(accu))
#             
#     mp.figure()
#     p1 = mp.plot(layers,avgAccu,'b--')
#     p2 = mp.plot(layers,avg_Err,'g*')
#     mp.legend((p1[0], p2[0]), ('Test', 'Train'))
#     mp.xticks(layers, ('2 layers', '4 layers','6 layers'))
#     mp.title('error vs depth of 40 nodes')
#     mp.xlabel('number of layers')
#     mp.ylabel('error')
#     
# # =============================================================================
# #     depth vs width
# # =============================================================================
#     models = [
#             [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1]}),
#              (Dense, [outputN], {'activation':'softmax', 'input_dim':20})
#              ],
#             [(Dense, [40], {'activation':'relu', 'input_dim':examples.shape[1]}),
#              (Dense, [outputN], {'activation':'softmax', 'input_dim':40})
#              ],
#          ]
# 
#     debug = False
#     c = []
#     if debug:
#         c.append(CrossValidator(examples, labels, models[2], epochs=50))
#     else:
#         for architecture in models:
#             c.append(CrossValidator(examples, labels, architecture, epochs=100))
#     
#     # do something useful with c... e.g. generate tables/graphs, etc.
#     avg_Err_Width = []
#     std_Err_Width = []
#     count = 0
#     
#     for a in c:
#         avg_Err_Width.append(np.average(a.get_errors()))
#         std_Err_Width.append(np.std(a.get_errors()))
#         count = count + 1
#     
#     x = np.arange(count)
#     print(avg_Err_Width)
#     
#     accu_width = []
#     for m in c:
#         model = m.models
#         for n in model:
#             y = n.evaluate(np.array(examples_test),np.array(labels_test),verbose = 0)
#             accu_width.append(1-y[1])
#             
#     models = [
#             [(Dense, [5], {'activation':'relu', 'input_dim':examples.shape[1]}),
#              (Dense, [5], {'activation':'relu', 'input_dim':5}),
#              (Dense, [10], {'activation':'relu', 'input_dim':5}),
#              (Dense, [outputN], {'activation':'softmax', 'input_dim':10})
#              ],
#             [(Dense, [10], {'activation':'relu', 'input_dim':examples.shape[1]}),
#              (Dense, [10], {'activation':'relu', 'input_dim':10}),
#              (Dense, [20], {'activation':'relu', 'input_dim':10}),
#              (Dense, [outputN], {'activation':'softmax', 'input_dim':20})
#              ]
#             ]
#             
#     c=[]
#     for architecture in models:
#         c.append(CrossValidator(examples, labels, architecture, epochs=100))
#         
#     # do something useful with c... e.g. generate tables/graphs, etc.
#     avg_Err_Depth = []
#     std_Err_Depth = []
#     count = 0
#     
#     for a in c:
#         avg_Err_Depth.append(np.average(a.get_errors()))
#         std_Err_Depth.append(np.std(a.get_errors()))
#         count = count + 1
#     
#     print(avg_Err_Depth)
#     
#     accu_Depth = []
#     for m in c:
#         model = m.models
#         y = model[1].evaluate(np.array(examples_test),np.array(labels_test),verbose = 0)
#         accu_Depth.append(1-y[1])
#         print("loss on test data", 1-y[1])
#     
#     mp.figure()
#     p2 = mp.plot(x,avg_Err_Width,'b--')
#     p4 = mp.plot(x,avg_Err_Depth,'g^')
#     mp.title('Width and Depth Analysis')
#     mp.xlabel('Number of nodes')
#     mp.ylabel('test error')
#     mp.legend(( p2[0], p4[0]), ( 'mean error of training data for shallow network','mean error of training data for deep network'))
#     mp.xticks(layers, ('20 nodes', '40 nodes'))
#     
# # =============================================================================
# #     explore alpha and l1
# # =============================================================================
# 
#     models.clear()
#     models = [
#         [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1], 
#           'kernel_regularizer': keras.regularizers.l1(0.1)}),
#          (Dense, [20], {'activation':'relu', 'input_dim':10,'kernel_regularizer': keras.regularizers.l1(0.1)}),
#          (Dense, [outputN], {'activation':'softmax', 'input_dim':20})
#         ],
#         [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1], 
#           'kernel_regularizer': keras.regularizers.l1(0.001)}),
#          (Dense, [20], {'activation':'relu', 'input_dim':10,'kernel_regularizer': keras.regularizers.l1(0.001)}),
#          (Dense, [outputN], {'activation':'softmax', 'input_dim':20})
#         ],
#         [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1], 
#           'kernel_regularizer': keras.regularizers.l1(0.0001)}),
#          (Dense, [20], {'activation':'relu', 'input_dim':10,'kernel_regularizer': keras.regularizers.l1(0.0001)}),
#          (Dense, [outputN], {'activation':'softmax', 'input_dim':20,})
#         ]
#         ]
#     
#     c=[]
#     for architecture in models:
#         c.append(CrossValidator(examples, labels, architecture, epochs=100))
#         
#     avg_Err_l1 = []
#     std_Err_l1 = []
#     count = 0
#     
#     for a in c:
#         avg_Err_l1.append(np.average(a.get_errors()))
#         std_Err_l1.append(np.std(a.get_errors()))
#         count = count + 1
#     
#     x = [0.1,0.001,0.0001]
#     mp.figure()
#     mp.plot(x,avg_Err_l1)
#     mp.title('L1 analysis')
#     mp.xlabel('alpha')
#     mp.ylabel('error')
#     
# # =============================================================================
# #     explore alpha and l2
# # =============================================================================
# 
#     models.clear()
#     models = [
#         [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1], 
#           'kernel_regularizer': keras.regularizers.l2(0.1)}),
#          (Dense, [20], {'activation':'relu', 'input_dim':10,'kernel_regularizer': keras.regularizers.l2(0.1)}),
#          (Dense, [outputN], {'activation':'softmax', 'input_dim':20})
#         ],
#         [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1], 
#           'kernel_regularizer': keras.regularizers.l1(0.001)}),
#          (Dense, [20], {'activation':'relu', 'input_dim':10,'kernel_regularizer': keras.regularizers.l2(0.001)}),
#          (Dense, [outputN], {'activation':'softmax', 'input_dim':20})
#         ],
#         [(Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1], 
#           'kernel_regularizer': keras.regularizers.l2(0.0001)}),
#          (Dense, [20], {'activation':'relu', 'input_dim':10,'kernel_regularizer': keras.regularizers.l2(0.0001)}),
#          (Dense, [outputN], {'activation':'softmax', 'input_dim':20,})
#         ]
#         ]
#     
#     c=[]
#     for architecture in models:
#         c.append(CrossValidator(examples, labels, architecture, epochs=100))
#         
#     avg_Err_l2 = []
#     std_Err_l2 = []
#     count = 0
#     
#     for a in c:
#         avg_Err_l2.append(np.average(a.get_errors()))
#         std_Err_l2.append(np.std(a.get_errors()))
#         count = count + 1
#     
#     x = [0.1,0.001,0.0001]
#     mp.figure()
#     mp.plot(x,avg_Err_l2)
#     mp.title('L2 analysis')
#     mp.xlabel('alpha')
#     mp.ylabel('error')
#     
# # =============================================================================
# #     explore dropout
# # =============================================================================
# 
#     models.clear()
#     models = [
#             [
#              (Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1]}),
#              (Dropout, [0.2], {}),(Dense, [20], {'activation':'relu', 'input_dim':20}),
#              (Dense, [outputN], {'activation':'softmax', 'input_dim':20})
#              ],
#             [
#              (Dense, [20], {'activation':'relu', 'input_dim':examples.shape[1]}),
#              (Dropout, [0.3], {}),
#              (Dense, [20], {'activation':'relu', 'input_dim':20}),
#              (Dense, [outputN], {'activation':'softmax', 'input_dim':20})
#              ]
#             ]
#     
#     c=[]
#     for architecture in models:
#         c.append(CrossValidator(examples, labels, architecture, epochs=100))
#         
#     # do something useful with c... e.g. generate tables/graphs, etc.
#     avg_Err_do = []
#     std_Err_do = []
#     count = 0
#     
#     for a in c:
#         avg_Err_do.append(np.average(a.get_errors()))
#         std_Err_do.append(np.std(a.get_errors()))
#         count = count + 1
#     
#     x = np.arange(count)
#     mp.figure()
#     mp.plot(x,avg_Err_do)
#     mp.title('drop out analysis')
#     mp.xlabel('alpha')
#     mp.ylabel('error')
#     
# =============================================================================
    
    
if __name__ == '__main__':
    
    
    main()
