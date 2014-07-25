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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



X = [ones(m,1) X];

size_y = size(y,1);

Y = zeros(size(y,1),num_labels);

for j = 1:size_y
	for k = 1:num_labels
		Y(j,k) = (y(j) == k);
	end
end


for i = 1:m
	sigm1 = sigmoid(Theta1*X(i,:)');
	sigm1 = [1; sigm1]; 
	sigm2 = sigmoid(Theta2*sigm1);
	J += -Y(i,:)*log(sigm2) - (1-Y(i,:))*log(1 - sigm2);	

end


regCost = sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2));
regCost = (regCost)*(lambda/2);

J += regCost;
J = J/m;




Delta1 = zeros(rows(Theta1_grad), columns(Theta1_grad));
Delta2 = zeros(columns(Theta2_grad), rows(Theta2_grad));

Delta2 = 0;
for i = 1:m
	
	a_1 = X(i,:);
	a_2 = sigmoid(Theta1*a_1');
	a_2 = [1; a_2];
	a_3 = sigmoid(Theta2*a_2);
	del_3 = a_3' - Y(i,:);
	z_2 = Theta1*a_1';
	z_2 = [1;z_2];
	del_2 = (del_3*Theta2)'.*sigmoidGradient(z_2);
	Delta2 = Delta2 + a_2*del_3;
	Delta1 = Delta1 + del_2(2:end)*a_1;
end

Theta2_grad = Delta2'/m;

Theta1_grad = Delta1/m;

tempReg1 = zeros(rows(Theta1_grad), columns(Theta1_grad));
tempReg1(:,2:end) = (lambda/m)*Theta1(:,2:end); 

tempReg2 = zeros(rows(Theta2_grad), columns(Theta2_grad));
tempReg2(:,2:end) = (lambda/m)*Theta2(:,2:end);

Theta2_grad += tempReg2;
Theta1_grad += tempReg1;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
