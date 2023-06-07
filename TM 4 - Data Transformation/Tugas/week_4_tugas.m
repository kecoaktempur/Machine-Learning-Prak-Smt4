sales = readtable("vgsales.csv");
sales_var = sales(:,{'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Global_Sales'});

% Mendeteksi missing value
if any(ismissing(sales_var))
    disp('Berikut adalah nilai yang missing');
    disp(sales_var(any(ismissing(sales_var),2),:));
else
    disp('Tidak terdapat nilai yang missing');
end

% Mengganti missing value dengan menghapusnya
missingVal = rmmissing(sales_var);
if any(ismissing(sales_var))
    disp('Berikut hasil missing value yang sudah diperbaiki:');
    disp(missingVal);
else
    disp('Tidak terdapat missing value');
end


outlier = detOutlier(table2array(missingVal));
if ~isempty(outlier)
    disp('Berikut adalah outlier');
    disp(outlier);
else
    disp('Tidak terdapat outlier');
end

% Mengganti outlier pada variabel input menggunakan mean
if ~isempty(outlier)
    disp('Penggantian outlier');
    disp(filloutliers(sales_var, 'nearest', 'mean'));
else 
    disp('Tidak ada outlier')
end

% Melakukan normalisasi pada variabel input
normalized = normalize(sales_var,"zscore");
disp('normalized')

% Function yang dapat mendeteksi outlier dengan Quartiles
function outlier = detOutlier(df)
    q1 = quantile(df, 0.25);
    q3 = quantile(df, 0.75);
    iqr = q3 - q1;
    outlier = df((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr)));
end