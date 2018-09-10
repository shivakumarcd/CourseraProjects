function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
      vals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30 ];
      %genrating all combinations for C and sigma
      cAndSigmas = zeros(64,2);;
      counter = 1;
      for i=1:8,
        for j=1:8,
          cAndSigmas(counter,1) = vals(i);
          cAndSigmas(counter,2) = vals(j);
          counter = counter + 1;
        end;
      end;
      
      
      %for debugging
      %cAndSigmas
      %fprintf('Break the code, testing combination generation\n');
      %pause;
      errorForEachCombination = ones(64);
      for counter = 1:64,
          %training on training set    
          % Train the SVM
          C = cAndSigmas(counter, 1);
          sigma = cAndSigmas(counter, 2);
          %fprintf('training for C=%f and sigma=%f', C, sigma);
          model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
          % Prediction on validation data
          pred = svmPredict(model, Xval);
          errorForEachCombination(counter) = mean(double(pred ~= yval));
      end;
      
      %fprintf("printing errorForEachCombination");
      %errorForEachCombination
      
      minError = min(min(errorForEachCombination));
      %minError
      rowIndex = find(errorForEachCombination == minError);
      %rowIndex
      
      C = cAndSigmas(rowIndex, 1);
      sigma = cAndSigmas(rowIndex, 2);
% =========================================================================

end
