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
* k Nearest neighbours
* Gaussian processes (with basic covariance functions like squared exponential).


## Examples

#### Linear regression with polynomial basis funtions

`regressor = LeastSquaresRegressor(scalar_polynomial, 4)
regressor.fit(X, y, visualise=True)`

#### Bayesian linear regression with polynomial basis functions

#### Logistic Regression with radial basis functions

#### Neural Network with sigmoid activation functions

#### Neural Network with relu activation functions
