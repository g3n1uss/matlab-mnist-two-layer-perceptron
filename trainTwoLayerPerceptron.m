function [hiddenWeights, outputWeights, ...
    error] = trainTwoLayerPerceptron(activationFunction, dActivationFunction,...
    numberOfHiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate)
% trainTwoLayerPerceptron Creates a two-layer perceptron and trains it on 
% the MNIST dataset.
%
% INPUT:
% activationFunction             : Activation function used in both layers.
% dActivationFunction            : Derivative of the activation
% function used in both layers.
% numberOfHiddenUnits            : Number of hidden units.
% inputValues                    : Input values for training (784 x 60000)
% targetValues                   : Target values for training (1 x 60000)
% epochs                         : Number of epochs to train.
% batchSize                      : Plot error after batchSize images.
% learningRate                   : Learning rate to apply.
%
% OUTPUT:
% hiddenWeights                  : Weights of the hidden layer.
% outputWeights                  : Weights of the output layer.
% 

    % The number of training vectors.
    trainingSetSize = size(inputValues, 2);
    
    % Input vector has 784 dimensions.
    inputDimensions = size(inputValues, 1);
    % We have to distinguish 10 digits.
    outputDimensions = size(targetValues, 1);
    
    % Initialize the weights for the hidden layer and the output layer.
    hiddenWeights = rand(numberOfHiddenUnits, inputDimensions);
    outputWeights = rand(outputDimensions, numberOfHiddenUnits);
    
    hiddenWeights = hiddenWeights./size(hiddenWeights, 2);
    outputWeights = outputWeights./size(outputWeights, 2);
    
    n = zeros(batchSize);
    
    figure('outerposition',[0 0 1024 600]); hold on;
    
    % IT SEEMS LIKE FOR EVERY EPOCH WE CHOOSE DATA RANDOMLY, CHANGE?

    for t = 1: epochs
        % Within each epoch choose 100 batches randomly
        for k = 1: batchSize
            % Randomly select which input vector to train on.
            % It might select the same vector several times, does it
            % matter? REWRITE?
            n(k) = floor(rand(1)*trainingSetSize + 1);
            
            % Propagate the input vector through the network.
            inputVector = inputValues(:, n(k));
            hiddenActualInput = hiddenWeights*inputVector;
            hiddenOutputVector = activationFunction(hiddenActualInput);
            outputActualInput = outputWeights*hiddenOutputVector;
            outputVector = activationFunction(outputActualInput);
            
            targetVector = targetValues(:, n(k));
            
            % Backpropagate the errors.
            outputDelta = dActivationFunction(outputActualInput).*(outputVector - targetVector);
            hiddenDelta = dActivationFunction(hiddenActualInput).*(outputWeights'*outputDelta);
            
            outputWeights = outputWeights - learningRate.*outputDelta*hiddenOutputVector';
            hiddenWeights = hiddenWeights - learningRate.*hiddenDelta*inputVector';
        end;
        
        % Calculate the error for plotting.
        error = 0;
        for k = 1: batchSize
            inputVector = inputValues(:, n(k));
            targetVector = targetValues(:, n(k));
            % Prediction. The following vector is real-valued, but we need
            % to convert it into the binary output. 1 indicates the number
            pred=activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector));
            % Convert to the binary vector
            [~,posMax]=max(pred);
            predInt=zeros(outputDimensions,1);
            predInt(posMax)=1;         
            %error = error + norm(pred - targetVector, 2);
            % Calculate error
            error = error + sum(abs(predInt - targetVector))/2;
        end;
        % Normalize error, the output is percentage
        error = error/batchSize*100;
        % Plot the error rate
        plot(t, error,'*')
        ylabel('Error, %')
        xlabel('Number of epochs')
        title(['Error rate of the two-layer perceptron with ', ...
            num2str(numberOfHiddenUnits),' hidden units and the learning rate '...
            , num2str(learningRate)])
    end;
end