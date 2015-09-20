# Polynomial_Regression

Goal of this project is to implement Polynomial Regression using Normal Equations and without using any Machine learning libraries

# Polynomial Regression using Normal Equation

Implemented Polynomial Regression for the below dataset

• Sinusoid Dataset: This is an artificial dataset created by randomly sampling the function
y = f(x) = 7.5 sin(2.25πx). You have a total of 100 samples of the input feature x and the
corresponding noise corrupted output/function values y. In addition to this you are also given
a validation set that has 50 samples. Since, there is only one input feature, I added higher powers
of the input feature p ∈ {1, 2, 3, . . . , 15} and calculated the RMSE on the validation set. 

• Yacht dataset: For this dataset, I added higher powers of the six input features, p ∈
{1, 2, 3, . . . , 7} and calculated the mean RMSE using ten fold cross validation. For example,
if p = 2 I had twelve features corresponding to the original six input features and
six new features obtained by squaring the original feature values.



