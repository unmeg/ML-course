function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % theta_0 = theta(1)
    % theta_1 = theta(2)

    % Can handle theta_0 and theta_1 without having to double handle the
    % values by making this code vectorised
    % Don't need to use sum or loops because we're summing via matrix 
    % multiplication (X' *) 

theta = theta - (alpha/m) * (X' * (X*theta - y));


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %debug
%     disp(J_history(iter))

end

end
