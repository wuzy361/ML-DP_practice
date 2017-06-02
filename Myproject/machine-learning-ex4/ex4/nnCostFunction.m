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
X =[ones(size(X,1),1),X];
hX1 = sigmoid(X*Theta1');
hX1 = [ones(size(hX1,1),1),hX1];
hX2 = sigmoid(hX1*Theta2');
c =[1:num_labels];
y = (y==c);
left = sum(sum(-y.*log(hX2)));
right = sum(sum((1-y).*log(1-hX2)));
unr = left-right;
J = 1/m*unr;

reg1 = sum(sum(Theta1(:,2:end).^2));
reg2 = sum(sum(Theta2(:,2:end).^2));
reg = lambda*(reg1+reg2)/(2*m);

J = J+reg;


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




for t = 1:m
  one_y = y(t,:)';
  a1= X(t,:);
  z2 = a1*Theta1';
  a2 = sigmoid(z2);
  a2 = [1 a2];
  z3 = a2*Theta2';
  a3 = sigmoid(z3');
  deta3 = a3 - one_y;
  z2 = [1 z2]; %����������õڶ��ַ����Ļ��Ͳ�����һ��;
  deta2 = Theta2' *deta3.*sigmoidGradient(z2');
  deta2 = deta2(2:end);
  
  
  Theta2_grad = Theta2_grad + deta3*a2;
  Theta1_grad = Theta1_grad + deta2*a1 ;
%------------------------------------------------------------------------------------
  %����ֻ��һ�����ز����������˵
  %���ǵڶ��ַ���������һ��û��ʲô��������֪ʶ����deta�Ķ��岻ͬ��
  %��һ�ַ�����deta�����sigmoid�ĵ������������ݶȾͲ��ó��ˡ�
  %�ڶ��ַ�����deta�Ȳ���sigmoid�ĵ������������ٳˣ����ַ���û�б�������ֻ�Ǽ�����Ⱥ�˳��ͬ��
  
  %���ڶ����˵��һ�ַ����ǶԵģ��ڶ���Ӧ���Ǵ�ġ�deta֮��Ĺ�ϵ�Ƿ����Եģ������Ա任���һ��f'(x)��
  %deta2 = ((Theta2')*deta3);
  %deta2 = deta2(2:end);
  
  
  %Theta2_grad = Theta2_grad + deta3*a2;
  %Theta1_grad = Theta1_grad + deta2.*sigmoidGradient(z2')*a1 ;  
%--------------------------------------------------------------------------------------  
end;

 Theta1_grad = Theta1_grad /m;
 Theta2_grad = Theta2_grad /m;

 


  
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
 bp_reg1 = lambda/m*Theta1;
 bp_reg2 = lambda/m*Theta2;
 bp_reg1(:,1) =0;
 bp_reg2(:,1) =0;
 Theta1_grad = Theta1_grad + bp_reg1;
 Theta2_grad = Theta2_grad + bp_reg2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
