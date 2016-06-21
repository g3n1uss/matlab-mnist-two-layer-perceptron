%applyTwoLayerPerceptronMNIST traina the two-layer perceptron using the 
% MNIST dataset and evaluates its performance.
    
% REWRITE IT AS A SCRIPT, WHY FUNCTION, IF IT HAS NO OUTPUT AND INPUT?

% Load MNIST data for training, the input contains 60000 images, 28x28 
% pixels each
% Each image is converted to a vector of length 784
% To summarize the input data 'inputData' is a matrix of 784x60000
inputData = loadMNISTImages('train-images.idx3-ubyte');
% Load the corresponding labels
% Each label indicated a number on the picture
% The labels are stored as a vector of length 60000
labels = loadMNISTLabels('train-labels.idx1-ubyte');

% Transform each labels to the binary format, for example, 
% 5 will be represented as [0 0 0 0 0 1 0 0 0 0]
targetData = 0.*ones(10, size(labels, 1));
for n = 1: size(labels, 1)
    targetData(labels(n) + 1, n) = 1;
end;

% Choose the number of hidden neurons:
numberOfHiddenUnits = 700;

% Choose the learning rate parameter
learningRate = 0.1;

% Choose activation function.
activationFunction = @logisticSigmoid;
dActivationFunction = @dLogisticSigmoid;

% Choose the batch size and the number of epochs. Remember there are 
% 60k input values
batchSize = 100;
epochs = 500;

fprintf('Train twolayer perceptron with %d hidden units.\n', numberOfHiddenUnits);
fprintf('Learning rate: %d.\n', learningRate);

[hiddenWeights, outputWeights, error] = trainTwoLayerPerceptron(activationFunction,...
    dActivationFunction, numberOfHiddenUnits, inputData, targetData, ...
    epochs, batchSize, learningRate);
% PRINT THE ERROR AFTER TRAINING
% Load validation set.
inputValues = loadMNISTImages('t10k-images.idx3-ubyte');
labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

% Choose decision rule.
fprintf('Validation:\n');
% REVIEW THE FOLLOWING FUNCTION, IT MIGHT BE INCORRECT
[correctlyClassified, classificationErrors] = validateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues, labels);

fprintf('Classification errors: %d\n', classificationErrors);
fprintf('Correctly classified: %d\n', correctlyClassified);