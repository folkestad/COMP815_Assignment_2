% clear all;
% close all;

% Load data
load('binaryalphadigs.mat')

% Reshape and format the input
data = dat(:);
X = zeros(36*39, 20*16);
for i = 1:size(data)
    pixels = reshape(data{i}, 1, []);
    for j = 1:size(pixels(:))
        X(i, j) = pixels(j);
    end
end

% Create one-hot encoded targets
T = zeros(36, 36*39);
value = 1;
for i = 1:36*39
    T(value, i) = 1;
    if mod(i, 39) == 0
        value = value + 1;
    end
end

% Transpose if needed
X = X';
T = T;

% Initiate the FFNN
neurons = 50000;
network = patternnet(neurons);

% Divide data into training, validataion and test sets
network.divideParam.trainRatio = 0.9;
network.divideParam.valRatio = 0.05;
network.divideParam.testRatio = 0.05;

% Train the FFNN
network = train(network,X,T);

% Get predictions from the FFNN
Y = network(X);

% Check performance of the FFNN
performance = perform(network,Y,T);







