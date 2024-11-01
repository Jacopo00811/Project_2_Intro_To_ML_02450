# from ExtractData import *
# from sklearn import model_selection
# import numpy as np

# Countrynames = np.unique(Country)
# # Load the Y data 
# y=df[Countrynames].values
# # initialize the attribute 
# # cross validation 
# Folds = 10
# outter_CV = model_selection.KFold(n_splits=Folds,shuffle=False)
# inner_CV = model_selection.KFold(n_splits=5,shuffle=False)
# currentfold=1
# #errors
# errors=[]
# #this is used for stat file
# y_pred_Baseline=np.zeros((len(y),len(Countrynames)))
# y_true=y
# #outerfold
# for train_index, test_index in outter_CV.split(X):
#     Data_train=y[train_index]
#     Data_test=y[test_index]
#     innerfolderrors=[]
#     #innerfold
#     for train_index, test_index in inner_CV.split(Data_train):
#         y_train = y[train_index,:]
#         y_test = y[test_index,:]
#         largestClass=y_train.sum(axis=0).argmax()
#         i=0
#         misclass=0
#         for i in range(len(y_test)):
#             if not ((y_test[i,largestClass] == 1)):
#                 misclass+=1
#         misclassRate=misclass/len(y_test)
#         innerfolderrors.append(misclassRate)
#     errors.append(min(innerfolderrors))

#     y_pred_Baseline[:,largestClass]=1
#     print(y_pred_Baseline.shape)
#     currentfold+=1
# print(errors)
              