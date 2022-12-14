load iris_dataset irisInputs irisTargets
average_acc = 0;
for i=[5 10 15 20]
accuracy = 0;
% repeating the experiment 10 times
for j=1:10
% creating the neural network with ith number of hidden layers
net = feedforwardnet(i);
net.trainParam.showWindow = 0;
% shuffling and dividing the dataset into train and test
net.divideParam.trainRatio = 60/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 20/100;
net.trainParam.epochs = 10*i;
net.trainParam.max_fail = 500;
% training the neural network
net = train(net, irisInputs, irisTargets);
% testing the neural network
outputs = net(irisInputs);
e = gsubtract(irisTargets,outputs);
% calculatingt the performance
performance = perform(net,irisTargets,outputs);
tind = vec2ind(irisTargets);
yind = vec2ind(outputs);
% calculating the percent errors
percentErr = sum(tind ~= yind)/numel(tind);
accu = 100 * (1 - percentErr);
accuracy = accuracy + accu;
end
accuracy = accuracy/10;
average_acc = average_acc + accuracy;
end
view(net)
average_acc = average_acc/4;
fprintf('Overall Average Accuracy = %.3f%%\n', average_acc);