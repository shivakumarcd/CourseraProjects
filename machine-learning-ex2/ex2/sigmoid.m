function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

      numOfRows = size(z, 1);
      numOfCols = size(z, 2);
      
      for i = 1:numOfRows,
        for j = 1:numOfCols,
          g(i, j) =  1/( 1 + exp( -(z(i, j)) )  );
        end;  
      end;

% =============================================================

end
