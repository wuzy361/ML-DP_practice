clear;
load('ex6data3.mat');
C = [0.01,0.03,0.1,0.3,1,3,10,30,]; sigma = [0.01,0.03,0.1,0.3,1,3,10,30,];
ans =0;
for i = 1:length(C)
  for j = 1:length(sigma)
    model= svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j))); 
    predict = svmPredict(model,Xval);
   
    if  mean(double(predict == yval)) > ans
       ans =  mean(double(predict == yval));
       printf("C = %d, sigma = %d, ans = %d \n",C(i),sigma(j),ans);
       
    endif;
   endfor;
endfor;