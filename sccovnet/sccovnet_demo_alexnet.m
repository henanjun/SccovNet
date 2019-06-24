
clear;clc;

dataset_name = 'NWPU45';  % NWPU45; AID30;WHU19;NWPU45; 
net = load('imagenet-caffe-alex.mat');
switch(dataset_name)
    case 'UCM21JPG'
       numClasses = 21;       
       train_ratio = 8;% 1-10;
    case 'AID30'
       numClasses = 30;
       train_ratio = 2;% 1-10;
    case 'NWPU45'
       numClasses = 45;
       train_ratio = 1;% 1-10;
end

opts.numClasses = numClasses;
opts.weightInitMethod = 'gaussian';
opts.lr = [ones(1,80)*0.001];
opts.batchSize =64;
opts.newlayerdecay = 1;
opts.newlayerlr = 1;

sfx = sprintf(['satgeone-batch[%d]-decay[%d]-lr[%d]0.0001-fc[73920,%d]'],...
         opts.batchSize, opts.newlayerdecay,opts.newlayerlr,numClasses) ;
opts.networkType = 'dagnn';
opts.numFetchThreads = 12;  
opts.lite = false ;
opts.train = struct() ;
opts.dataset_name = dataset_name; 
if ~isfield(opts.train, 'gpus'), opts.train.gpus = [1]; end



% -------------------------------------------------------------------------
%Prepare data
% -------------------------------------------------------------------------

checkpoint = 'checkpoint';
datapath = 'data';


for dataset_index=1
%aul_name = ['imdb',dataset_name,'_',num2str(train_ratio),'.mat'];
if train_ratio<1
    train_ratio = train_ratio*10;
end
aul_name = [datapath,'/','imdb',dataset_name,'_',num2str(train_ratio),'_',num2str(dataset_index),'.mat'];
if ~exist(aul_name) 
    imdb_all_name = [datapath,'/','imdb_all',dataset_name];
    train_ratio = train_ratio/10;
    load(imdb_all_name);
    [imdb] = fun_RandomSplit(imdb_all,train_ratio,dataset_name);
    images = imdb.images.name(find(imdb.images.set == 1)) ;
    [averageImage, rgbMean, rgbCovariance] = getImageStats(images, ...
                                                    'imageSize', [256 256], ...
                                                    'numThreads', opts.numFetchThreads, ...
                                                    'gpus', opts.train.gpus) ;
    [v,d] = eig(rgbCovariance) ;
    rgbDeviation = v*sqrt(d) ;
    clear v d 
    save(aul_name,'imdb','averageImage','rgbMean','rgbCovariance','rgbDeviation')
else 
end
% end:prepara data
%------------------------------------------------------------------------------
opts.expDir = fullfile('data', [dataset_name '-' sfx],['imdb',dataset_name,'_',num2str(train_ratio),'_',num2str(dataset_index)]) ;


sccovnet_func_alexnet(net,aul_name,opts)
end