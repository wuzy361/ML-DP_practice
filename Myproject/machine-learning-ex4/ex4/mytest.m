
clear;
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   

load('ex4data1.mat');
load('ex4weights.mat');

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];
lambda = 1;
%c =[1:10];
%y= (y==c);
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
                   
%y(1:10,:)*log(hX2(1:10,:))

%sum(log(hX2(1:10,:)))