alpha =0.01
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
size(theta)
size(alpha*(1/m*((X*theta-y)'*X)))
theta = theta - alpha*(1/m*((X*theta-y)'*X))';
size(theta)
computeCost(X, y, theta);
figure;
plot(X(:,2),y,'rx','MarkerSize',10);
hold;
plot(X(:,2), X*theta, '-')
