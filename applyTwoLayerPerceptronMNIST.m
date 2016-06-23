%applyTwoLayerPerceptronMNIST traina the two-layer perceptron using the 
% MNIST dataset and evaluates its performance.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assign the default values to the parameters %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Choose the number of hidden neurons:
numberOfHiddenUnits = 10;
% Increasing the number of hidden neurons almost does not affect the performance

% Choose the learning rate parameter
learningRate = 0.1;

% Activation function and its derivative can be specified in 'activationFuncs.m'.
actFuncs=activationFuncs;
activationFunction = actFuncs.activationFunc;
dActivationFunction = actFuncs.dActivationFunc;

% Choose the batch size and the number of epochs. Remember there are 
% 60k input values
batchSize = 100;
epochs = 500;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load and transform the data                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath(genpath('data'));

% Load MNIST data for training, the input contains 60000 images, 28x28 
% pixels each
% Each image is converted to a vector of length 784
% To summarize the input data 'inputData' is a matrix of 784x60000
inputData = loadMNISTImages('data/train-images.idx3-ubyte');
% Load the corresponding labels
% Each label indicated a number on the picture
% The labels are stored as a vector of length 60000
labels = loadMNISTLabels('data/train-labels.idx1-ubyte');

% Transform each labels to the binary format, for example, 
% 5 will be represented as [0 0 0 0 0 1 0 0 0 0]
targetData = 0.*ones(10, size(labels, 1));
for n = 1: size(labels, 1)
    targetData(labels(n) + 1, n) = 1;
end;

fprintf('Train twolayer perceptron with %d hidden units.\n', numberOfHiddenUnits);
fprintf('Learning rate: %d.\n', learningRate);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train the two layer perceptron              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[hiddenWeights, outputWeights, error] = trainTwoLayerPerceptron(activationFunction,...
    dActivationFunction, numberOfHiddenUnits, inputData, targetData, ...
    epochs, batchSize, learningRate);

fprintf('Error: %d%%.\n', error);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validation of the two layer perceptron      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load validation set.
inputValues = loadMNISTImages('data/t10k-images.idx3-ubyte');
labels = loadMNISTLabels('data/t10k-labels.idx1-ubyte');

% Choose decision rule.
fprintf('Validation:\n');
% REVIEW THE FOLLOWING FUNCTION, IT MIGHT BE INCORRECT
[correctlyClassified, classificationErrors] = validateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues, labels);

fprintf('Classification errors: %d\n', classificationErrors);
fprintf('Correctly classified: %d\n', correctlyClassified);
