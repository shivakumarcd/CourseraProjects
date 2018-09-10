function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

      onesVector = ones(m, 1);
      y = -1 * y;
      z = X * theta;
      
      J = (1/m) * sum(  (y .* log(sigmoid(X * theta)))  -  ((onesVector + y) .* log(onesVector - sigmoid(X * theta)))  );

      %hell!! almost use y without converting it to its original value
      %therefor always appropriately named temp variables in future, like negativeY in this case
      y = -1 * y;
      numOfRows = length(grad);
      for j=1:numOfRows,
          grad(j, 1) = (1/m) * sum( (sigmoid(X * theta) - y).* X(:, j) );
      end;
% =============================================================

end
