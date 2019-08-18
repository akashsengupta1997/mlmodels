# mlmodels
Numpy implementations of basic ML algorithms.

Currently implemented:  
* Linear regression (least squares)
* Bayesian linear regression
* Logistic regression
* Neural networks

Also includes polynomial and radial basis functions.

TODO:
* k-means
* k-NN (+ PCA for low dimensional embedding before k-NN)
* Gaussian processes (with basic covariance functions like squared exponential).


## Examples

#### Linear regression with polynomial basis funtions

`regressor = LeastSquaresRegressor(scalar_polynomial, 4)`  
`regressor.fit(X, y, visualise=True)`

<img src="https://github.com/akashsengupta1997/mlmodels/blob/master/linreg.png" alt="drawing" width="350"/>

#### Bayesian linear regression with polynomial basis functions

`bayes_regressor = BayesianLinearRegressor(0.1, noise_var, scalar_polynomial, 4)`  
`bayes_regressor.compute_posterior(X, y, visualise=True)`

<img src="https://github.com/akashsengupta1997/mlmodels/blob/master/bayesian_reg.png" alt="drawing" width="350"/>


#### Logistic Regression with radial basis functions

`log_res = LogisticRegressor(gaussian_rbf, X, 0.2)`  
`log_res.fit(X_train, y_train, 0.005, 100, 64, X_val, y_val, visualise_training=True)`
<img src="https://github.com/akashsengupta1997/mlmodels/blob/master/logres_surface.png?raw=true" alt="drawing" width="350"/>
<img src="https://github.com/akashsengupta1997/mlmodels/blob/master/logres_contour.png?raw=true" alt="drawing" width="350"/>

#### Neural Network with sigmoid activation functions

`nn = NeuralNetwork(X.shape[1], [8, 8], ['sigmoid', 'sigmoid'], 'sigmoid')`  
`nn.fit(X_train, y_train, 0.005, 8000, 16, X_val, y_val, 16, 'binary_crossentropy',  
       visualise_training=True, save_name=None, display_metrics=True, compute_accuracy=True)`
<img src="https://github.com/akashsengupta1997/mlmodels/blob/master/nn_sigmoid_surface.png" alt="drawing" width="350"/>
<img src="https://github.com/akashsengupta1997/mlmodels/blob/master/nn_sigmoid_contour.png" alt="drawing" width="350"/>


#### Neural Network with relu activation functions

`nn = NeuralNetwork(X.shape[1], [8, 8], ['relu', 'relu'], 'sigmoid')`  
`nn.fit(X_train, y_train, 0.005, 8000, 16, X_val, y_val, 16, 'binary_crossentropy',  
       visualise_training=True, save_name=None, display_metrics=True, compute_accuracy=True)`
<img src="https://github.com/akashsengupta1997/mlmodels/blob/master/nn_relu_surface.png" alt="drawing" width="350"/>
<img src="https://github.com/akashsengupta1997/mlmodels/blob/master/nn_relu_contour.png" alt="drawing" width="350"/>

