function funcs = activationFuncs
%ACTIVATIONFUNCS Contains the activation function and its derivative
funcs.activationFunc=@activationFunc;
funcs.dActivationFunc=@dActivationFunc;

end
% Logistic sigmoid activation function
function y=activationFunc(x)
 y = 1./(1 + exp(-x)); % 5% error

% relu
% y=max(0,x); % terrible performance

% softsign
%y = x./(1+abs(x)); % 12% error with softsign

% arctanh
% y=atanh(x); % terrible performance

% tanh
% y=tanh(x); % error 18%

% sin(x)
% y=sin(x); % terrible performance
end

% Derivative of the activation function
function y=dActivationFunc(x)
 y = activationFunc(x).*(1 - activationFunc(x));

% relu
%  [len,~]=size(x);
%  y=zeros(len, 1);
%  for i=1:len
%      if x(i,1)<=0
%          y(i,1)=0;
%      else
%          y(i,1)=1;
%      end
%  end

% softsign
%y=1./(1+abs(x.^2));

% arctanh
% y=1./(1+x.^2);

% tanh
% y=1-activationFunc(x).^2;

% sin(x)
% y=cos(x);
end
