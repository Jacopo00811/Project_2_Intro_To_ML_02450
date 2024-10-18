from ExtractData import *
import matplotlib.pyplot as plt
from sklearn import model_selection, linear_model as lm
from dtuimldmtools import rlr_validate #train_neural_net, draw_neural_net, visualize_decision_boundary


# print(df.head())
# Extract target and features
y = df['Price (in USD)'].to_numpy()
X = df.drop(['Price (in USD)', 'Car Make', 'Country'], axis=1).to_numpy()

# Handle NaNs and convert to float64
X = np.nan_to_num(X).astype(float)

# Add bias term (offset)
X = np.hstack([np.ones((X.shape[0], 1)), X])

attributeNames = ['Offset', 'Year', 'Horsepower', 'Torque (lb-ft)', '0-60 MPH Time (seconds)']

N, M = X.shape




## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)

lambdas = np.array([0, 10**-3, 10**-2, 10**-1, 10**1, 10**2, 10**3, 10**4, 10**5], dtype=float)


Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))


k=0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    # Avoid division by zero by replacing 0 std with a small epsilon
    epsilon = 1e-8  # Small constant to prevent divide-by-zero errors
    safe_sigma = np.where(sigma[k, :] == 0, epsilon, sigma[k, :])

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / safe_sigma
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / safe_sigma
    # print(np.isnan(X_train).any(), np.isnan(X_test).any())

    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term


    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    # m = lm.LinearRegression().fit(X_train, y_train)
    # Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    # Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K-1:
        plt.figure(k, figsize=(14,8))
        plt.subplot(1,2,1)
        plt.semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        plt.xlabel('Regularization factor', fontweight="bold")
        plt.ylabel('Mean Coefficient Values', fontweight="bold")
        plt.title('Regularization factor vs Mean Coefficient Values', fontweight="bold")
        plt.grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        plt.legend(attributeNames[1:], loc='best')
        
        plt.subplot(1,2,2)
        plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)), fontweight="bold")
        plt.loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        plt.xlabel('Regularization factor', fontweight="bold")
        plt.ylabel('Squared error (crossvalidation)', fontweight="bold")
        plt.legend(['Train error','Validation error'])
        plt.grid()
        plt.suptitle("Regularization with Cross-Validation", fontsize=20, fontweight="bold", color="red")
        plt.show()
    
    # To inspect the used indices, use these print statements
    # print('Cross validation fold {0}/{1}:'.format(k+1,K))
    # print('Train indices: {0}'.format(train_index))
    # print('Test indices: {0}\n'.format(test_index))

    k+=1


# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(len(attributeNames)):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1], 2))) 