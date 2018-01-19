function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
% test = X;
% thing = size(X);
% columns = thing(2);
% for i=1:columns
%     X_use = X(:,i);
%     mu = mean(X_use);
%     sigma = std(X_use);
%     test = (X_use-mu)/sigma
% end

mu = mean(X);
sigma = std(X);
mu_vector = ones(length(X),1) * mu;
sigma_vector = ones(length(X),1) * sigma;

X_norm = X - mu_vector ./ sigma_vector;


% len = length(X);
% mu = mean(X);
% sigma = std(X);
% X_norm = (X - ones(len, 1) * mu) ./ (ones(len, 1) * sigma);
% 


 disp(X_norm)
% ============================================================

end
