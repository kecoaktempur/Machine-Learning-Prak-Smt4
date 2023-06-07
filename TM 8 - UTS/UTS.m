data = readtable('2_heart.csv');
disp(data)

%Check Outlier
checkOutlier = sum(isoutlier(data))

%Check Missing
checkMissing = sum(ismissing(data))

%Fill Outlier
fixData = filloutliers(data,'nearest','mean');%mengganti data outlier dengan data mean terdekat darj data D
disp(fixData);

%Normalisasi Min Max
for i = 1:14
    for j = 1:length(fixData{:,i})
        dataA = fixData(:,i);
        dataB = table2array(dataA);
        dataNormal(j,i) =(fixData{j,i}-min(dataB))/(max(dataB)-min(dataB));
    end
end
disp("Data Normal");
disp(dataNormal);

%Split data
dataSplit = cvpartition(size(dataNormal,1),'HoldOut',0.3);
index = dataSplit.test;

%Data Training dan Data Test
dataTrain = data(~index,:);
dataTest  = data(index,:);

%Input Data Train
dataTrainY = dataTrain(:, 14);
dataTrainX = dataTrain(:, 1:13);

%Input data Test
dataTestY = dataTest(:, 14);
dataTestX = dataTest(:, 1:13);

%Klasifikasi Data Tree
dataKlasifikasi = fitctree(dataTrainX, dataTrainY);
predictY = predict(dataKlasifikasi, dataTestX);
view(dataKlasifikasi, 'Mode', 'graph');

%Confusion Chart
chart = confusionmat(table2array(dataTestY), predictY);
confusionchart(chart);
accuracy = 100*sum(diag(chart))./sum(chart(:))


