function A_inv_b = matrixInverseVector(A, b, x_init, alpha)
cost =(norm(A*x_init-b))^2;
%disp(sprintf('init cost = %f\n',cost))
  while cost >=10^-8,
    x_init = x_init - alpha*(2*A*(A*x_init-b));
    cost =(norm(A*x_init-b))^2;
    %disp(sprintf('cost = %f\n',cost))
  end;
  A_inv_b = x_init
endfunction