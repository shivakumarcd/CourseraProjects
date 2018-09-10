function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
    Xt = X';
    sizeXt = size(Xt);
    numRowsInXt = sizeXt(1,1);
    numColsInXt = sizeXt(1,2);
    Xt = [ones(1,numColsInXt); Xt];
    
    % size(Xt)
    % gave 401*5000
    
    Jsum = 0;
    for j=1:numColsInXt,
      activationColumn = Xt(:, j);
      activationColumn = Theta1 * activationColumn;
      activationColumn = sigmoid(activationColumn);
      activationColumn = [1; activationColumn];
      activationColumn = Theta2 * activationColumn;
      activationColumn = sigmoid(activationColumn);
      
      actualNumber = y(j,1);
      actualY = zeros(num_labels,1);
      actualY(actualNumber, 1) = 1;
      
      %now we have g(h(theta)) == g(z) == a in activationColumn
      %now we have y in actualY
      %now applying formula of J
      tmpVectorOfK = -actualY .* log(activationColumn) - (1-actualY) .* log(1-activationColumn);
      Jsum = Jsum + sum(tmpVectorOfK); %later will be divided by m as this is summation over m examples
    end;
    
    
    %now adding regularization factor
    tmpSum = 0;
    
    sizeOfTheta1 = size(Theta1);
    for i=1:sizeOfTheta1(1,1),
      for j=2:sizeOfTheta1(1,2),
        tmpSum = tmpSum + Theta1(i,j) * Theta1(i,j);    
      end;
    end;
    
    sizeOfTheta2 = size(Theta2);
    for i=1:sizeOfTheta2(1,1),
      for j=2:sizeOfTheta2(1,2),
        tmpSum = tmpSum + Theta2(i,j) * Theta2(i,j);    
      end;
    end;
    
    %combining two results
    J = Jsum/m + ( (lambda*tmpSum) / (2 * m)) ;
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
      Xt = X';
      sizeXt = size(Xt);
      numRowsInXt = sizeXt(1,1);
      numColsInXt = sizeXt(1,2);
      Xt = [ones(1,numColsInXt); Xt];
      
      for j=1:numColsInXt,
        activationColumn1 = Xt(:,j);
        
        activationColumn2 = Theta1 * activationColumn1;
        activationColumn2 = sigmoid(activationColumn2);
        activationColumn2 = [1; activationColumn2];
        
        activationColumn3 = Theta2 * activationColumn2;
        activationColumn3 = sigmoid(activationColumn3);
        
        actualNumber = y(j,1);
        actualY = zeros(num_labels,1);
        actualY(actualNumber,1) = 1;
        
        %now we have g(h(theta)) == g(z) == a3 in activationColumn3
        %now we have y in actualY
        %calculating deltaS
        delta3 = activationColumn3 - actualY;
        
        %calculating delta2
        delta2 = Theta2' * delta3 .* (activationColumn2 .* (1-activationColumn2));
        
        %now calculating Theta1grad and Theta2grad
        Theta1_grad = Theta1_grad + (delta2(2:end) * activationColumn1');
        Theta2_grad = Theta2_grad + (delta3 * activationColumn2');
        
        %Theta1_grad = Theta1_grad + (delta2(2:end) * (activationColumn1(2:end))');
        %Theta2_grad = Theta2_grad + (delta3(2:end) * (activationColumn2(2:end))');
        
      end;
      Theta1_grad = Theta1_grad/m;
      Theta2_grad = Theta2_grad/m;
    
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
      numRows = size(Theta1_grad,1);
      numCols = size(Theta1_grad,2);
      for i=1:numRows,
        for j=2:numCols,
          Theta1_grad(i,j) = Theta1_grad(i,j) + (lambda/m)*Theta1(i,j);
        end;
      end;
        
      numRows = size(Theta2_grad,1);
      numCols = size(Theta2_grad,2);
      for i=1:numRows,
        for j=2:numCols,
          Theta2_grad(i,j) = Theta2_grad(i,j) + (lambda/m)*Theta2(i,j);
        end;
      end;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
