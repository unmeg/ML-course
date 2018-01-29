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

% add bias
bias = ones(m, 1);
x_biased = [bias, X];

% roll out our classes
I = eye(num_labels);
y_spread = I(y,:);

% massage thetas
% no bias
tm1 = Theta1(:,2:end);
tm2 = Theta2(:,2:end);

% forward prop
z2 = x_biased * Theta1';
a2 = [bias sigmoid(z2)]; % add bias

z3 = a2 * Theta2';
a3 = sigmoid(z3);

% get output error
d3 = a3 - y_spread;

% fprintf(['Should be size of a3 and y.. Next 3 values should be same'])
% size(a3)
% size(y_spread)
% size(d3) 

% z2 should be m * n . n * h -> m x h
% 
% fprintf(['Should be size of a1 height and theta width .. '])
% size(x_biased)
% size(Theta1)
% size(z2) 

% Theta2_nobias = Theta2(:,2:end); % remove bias
% d2 = d3 * Theta2_nobias .* sigmoidGradient(z2); why does this wig out?
d2_hold = d3 * tm2;
d2 = d2_hold .* sigmoidGradient(z2);

% fprintf(['Should be same size .. '])
% size(z2)
% size(d2)

% Create our delta matrices
DELTA_1 = zeros(size(Theta1));
DELTA_2 = zeros(size(Theta2));

% Do things to them
% DELTA_1_COMPARE = x_biased' * d2;
DELTA_1 = DELTA_1 + d2' * x_biased;
% DELTA_2 = a2' * d3;
DELTA_2 = DELTA_2 + d3' * a2;

% gimme a vector of zeros -- will use this to account for when j = 0 
insert_zero_1 = [ zeros(size(Theta1, 1), 1), tm1];
insert_zero_2 = [ zeros(size(Theta2, 1), 1), tm2];

% calc
Theta1_grad = (1/m)*(DELTA_1 + (lambda/m * insert_zero_1));
Theta2_grad = (1/m)*(DELTA_2 + (lambda/m * insert_zero_2));

% Cost
J = (-1/m)*(sum(sum((y_spread .* log(a3)  + (1 - y_spread) .* log(1-a3)),2)));

% regularise
t1 = sum(sum(tm1.^2));
t2 = sum(sum(tm2.^2));

r = (lambda / (2*m)) * (t1 + t2)

J = J + r;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
