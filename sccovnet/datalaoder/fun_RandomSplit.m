function [imdb] = fun_RandomSplit(imdb_all,train_ratio,dataset_name)
% Random Split input (allsamples_name) into training and test set
% Note: this function may be not suitable for large data set
allsamples_name = imdb_all.allsamples_name;
perclass_name = imdb_all.class_name;
switch(dataset_name)  % UCM21; AID30;WHU19;NWPU45;
    case 'UCM21JPG'
        perclass_total = ones(1,21)*100;% 21 classes (each class contains 100 images)
    case 'AID30'
        perclass_total = [360,310,220,400,360,260,240,350,410,300,...
                                              370,250,390,280,290,340,350,390,370,420,...
                                              380,260,290,410,300,300,330,290,360,420] ;
    case 'NWPU45'
        perclass_total = ones(1,45)*700;
end

no_classes = numel(perclass_total);
perclass_train = ceil(perclass_total*train_ratio);
gt_labels_all = [];
for i = 1:no_classes
    gt_labels_all = [gt_labels_all;ones(perclass_total(i),1)*i];
end

[train_SL,test_SL,perclass_test] = GenerateSample(gt_labels_all,perclass_train,no_classes);

imdb.images.id = 1:numel(allsamples_name);

imdb.images.set(1:sum(perclass_train))=1; % train set 
imdb.images.set(sum(perclass_train)+1:sum(perclass_total))=2;%val(test) set

imdb.images.label(1:sum(perclass_train))=train_SL(2,:);
imdb.images.label(sum(perclass_train)+1:sum(perclass_total))=test_SL(2,:);

imdb.images.name(1:sum(perclass_train))= allsamples_name(train_SL(1,:));% roots of training images
imdb.images.name(sum(perclass_train)+1:sum(perclass_total))=allsamples_name(test_SL(1,:));% roots of test images

imdb.classes.description = perclass_name;
imdb.classes.name = perclass_name;
end