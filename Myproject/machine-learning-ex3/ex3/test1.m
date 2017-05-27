clc;clear;
load('ex3data1.mat');


rand_indices = randperm(size(X,1));
trainX = X(rand_indices(1:4000), :);
trainy = y(rand_indices(1:4000));

testX = X(rand_indices(4000:5000), :);
testy = y(rand_indices(4000:5000));
testX = [ones(1001, 1) testX];

X = trainX;
y= trainy;


m = size(X, 1);
n = size(X, 2);
%迭代次数
iternum = 1000;
all_theta = (rand(10, n + 1)/4000);
%all_theta = ones(10,n+1)/4000;
all_theta = all_theta';
% Add ones to the X data matrix
X = [ones(m, 1) X];
savex = [1:iternum];
savey = zeros(1,iternum);
cnt =0;
c = [1:10];

%lr是学习率，lambda是正则化系数。
lr = 0.01;
lambda = 0.1;

for i = 1:iternum
[J, grad,h,re] = lrCostFunction(all_theta, X, y==c, lambda);
  sum(sum(J));
  savey(i) = sum(sum(J));
  cnt = cnt+1;
  all_theta = all_theta - lr*grad;
end;
[val,indx]=min(savey)
plot(savex,savey)
%
pre =sigmoid(testX * all_theta);
[val,predict] = max(pre');
acc = sum(predict == testy')/1000


