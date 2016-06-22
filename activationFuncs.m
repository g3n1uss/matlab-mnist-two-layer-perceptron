function funcs = activationFuncs
%ACTIVATIONFUNCS Contains the activation function and its derivative
funcs.activationFunc=@activationFunc;
funcs.dActivationFunc=@dActivationFunc;

end
% Logistic sigmoid activation function
function y=activationFunc(x)
y = 1./(1 + exp(-x));
end

% Derivative of the activation function
function y=dActivationFunc(x)
y = activationFunc(x).*(1 - activationFunc(x));
end
