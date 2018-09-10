function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

      for rowIndex = 1:m,
        trainingExample = (X(rowIndex,:))';
        trainingExample = [1; trainingExample];
        %computing first layer
        layer2Matrix = Theta1 * trainingExample;
        %computing sigmoid for each value
        for rowIndex2 = 1:length(layer2Matrix),
          layer2Matrix(rowIndex2,1) = sigmoid(layer2Matrix(rowIndex2,1));
        end;
        %suffixing 26th element
        layer2Matrix =  [1; layer2Matrix];
        
        %computing 2nd layer
        layer3Matrix = Theta2 * layer2Matrix;
        %computing sigmoid for each value
        for rowIndex2 = 1:length(layer3Matrix),
          layer3Matrix(rowIndex2,1) = sigmoid(layer3Matrix(rowIndex2,1));
        end;
        
        
        %now finding index of max element
        [maxRowIndex, maxColIndex] = find(layer3Matrix==(max(layer3Matrix)));
        
        %p(rowIndex, 1) = maxRowIndex;
        
        if(maxRowIndex == 10),
          p(rowIndex, 1) = 0;
        else
          p(rowIndex, 1) = maxRowIndex;
        end
      end;







% =========================================================================


end
