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
x1 = [1 2 1]; x2 = [0 4 -1];
Cvec = [0.01 0.03 0.1 0.3 1 3 10 30];
Svec = [0.01 0.03 0.1 0.3 1 3 10 30];
predErrors = zeros(length(Cvec) * length(Svec), 3); % [Cval, Sval, ErrVal] for each iter

for i = 1:length(Cvec)
  for j = 1:length(Svec)
    % train the SVM using C and sigma combination
    model= svmTrain(X, y, Cvec(i), @(x1, x2) gaussianKernel(x1, x2, Svec(j)));
    %compute predictions based on the model
    predictions = svmPredict(model, Xval);
    %compute prediction errors
    predError = mean(double(predictions ~= yval));
    %fill in the prediction errors matrix
    predErrors(length(Cvec)*(i-1)+j, :) = [Cvec(i) Svec(j) predError];
   end
end

%find the minimum error in the prediction errors matrix
[x, ix] = min(predErrors(:, 3));
%fprintf('Minimum prediction error = %f at matrix index %d\n', x, ix);
C = predErrors(ix, 1);
sigma = predErrors(ix, 2);
%fprintf('Choosing C = %d, and sigma = %f', C, sigma);



% =========================================================================

end
