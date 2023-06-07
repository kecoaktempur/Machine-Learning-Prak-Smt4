data = readtable('2_heart.csv');
disp(data)

% Preprocessing
disp('Preprocessing')
checkNa = ismissing(data)
cleanData = rmmissing(data)

% Normalisasi
normalizedData = normalize(cleanData,"zscore")

matrixData = table2array(normalizedData)
Y = matrixData(:,11)

%PCA
[coeff, score, ~,~,explained] = pca(matrixData)

%Check Outlier
checkOutlier = sum(isoutlier(data))

%Check Missing
checkMissing = sum(ismissing(data))

%Fill Outlier
fixData = filloutliers(data,'nearest','mean') %mengganti data outlier dengan data mean terdekat darj data D
disp(fixData)

%Split data
dataSplit = cvpartition(size(normalizedData,1),'HoldOut',0.3);
index = dataSplit.test

%Data Training dan Data Test
dataTrain = data(~index,:);
dataTest  = data(index,:);

%Input Data Train
dataTrainY = dataTrain(:, 14);
dataTrainX = dataTrain(:, 1:13);

%Input data Test
dataTestY = dataTest(:, 14);
dataTestX = dataTest(:, 1:13);

%Merubah table menjadi double array
dataTestYNew = table2array(dataTestY);

% Decission Tree
% Train classifier on training data
dt = fitctree(dataTrainX, dataTrainY)

% Test classifier on testing data 
predictYdt = dt.predict(dataTestX)

% Evaluate performance
cmdt = confusionmat(dataTestYNew,predictYdt)
confusionchart(cmdt)
accuracydt = sum(predictYdt == dataTestYNew)/length(dataTestYNew)

% Naive Bayes
% Train classifier on training data 
nb = fitcnb(dataTrainX, dataTrainY)

% Test classifier on testing data 
predictYnb = nb.predict(dataTestX)

% Evaluate performance
cmnb = confusionmat(dataTestYNew, predictYnb)
confusionchart(cmnb)
accuracynb = sum(predictYnb == dataTestYNew)/length(dataTestYNew)

% Random Forest
% Train classifier on training data
nTrees = 100; % set the number of trees in the random forest
rng(1); % set the random seed for reproducibility
rf = TreeBagger(nTrees, dataTrainX, dataTrainY, 'Method', 'classification')

% Test classifier on testing data 
predictYrf = rf.predict(dataTestX);
predictYrf = str2double(predictYrf) % convert predicted labels from string to numeric

% Evaluate performance
cmrf = confusionmat(dataTestYNew, predictYrf)
confusionchart(cmrf)
accuracyrf = sum(predictYrf == dataTestYNew)/length(dataTestYNew)