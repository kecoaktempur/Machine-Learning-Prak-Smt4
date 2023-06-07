A = readtable('medMentalHealth.csv');
Outlier_data1 = isoutlier(A);
fill_data1 = filloutlier(A, 0);
rm_data1= rmoutliers(A);

fill_missing = ismissing(fill_data1);
rm_missing = ismissing(rm_data1);