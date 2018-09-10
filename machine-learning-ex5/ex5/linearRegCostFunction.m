function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

  %calculating cost
  hTheta = X * theta;
  J = sum((hTheta-y) .* (hTheta-y));
  J = J/(2*m);
 
  thetaSq = theta .* theta;
  regTerm = sum(thetaSq(:)) - thetaSq(1,1);
  regTerm = (lambda * regTerm) /(2*m);

  J = J + regTerm;
  
  %calculating gradients
  summation = 0;
  for j=1:size(grad,1),
    if j==1,
      summation = sum(hTheta-y);
    else
      summation = sum(X(:,j).*(hTheta-y)) + lambda*theta(j,1);
    end
    summation = summation/m;
    grad(j,1) = summation;
  end;
% =========================================================================
grad = grad(:);

end
