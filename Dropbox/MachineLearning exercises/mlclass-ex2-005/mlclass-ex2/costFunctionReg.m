function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
tempJ = -y.*log(sigmoid(X*theta));
tempJ += -(1-y).*log(1-sigmoid(X*theta));

sumJ = sum(tempJ);
thetaSum = 0;

for i= 2:size(theta)
	thetaSum += theta(i)^2;
endfor

thetaSq = lambda * thetaSum /2;
J = sumJ + thetaSq;
J = J/m;

for i = 1:m
	grad(1) += (sigmoid(X(i, :)*theta)-y(i))*X(i,1);
endfor

for j = 2:size(theta)
        for i = 1:m
                grad(j) += (sigmoid(X(i, :)*theta)-y(i))*X(i,j);
        endfor
	grad(j) += lambda*theta(j);
endfor

grad = grad./m;





% =============================================================

end
