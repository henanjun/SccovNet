

function sccovnet_func_alexnet(net,datasetname,opts)

load(datasetname);
opts.newlayerdecay = 1;
opts.newlayerlr = 1;
opts.averageImage = rgbMean;
opts.classNames = imdb.classes.name;
opts.weightInitMethod = 'gaussian';
opts.scale = 1;
net.layers(14:end)=[];
opts.initilizeFromLR  = true;
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ; 

name = 'conv3_reduced';
net.addLayer(name,dagnn.ChannelWisePooling('ksize',6,'stride',6),{'x9'},{name});

name = 'conv4_reduced';
net.addLayer(name,dagnn.ChannelWisePooling('ksize',6,'stride',6),{'x11'},{name}); 

name = 'conv5_reduced';
net.addLayer(name,dagnn.ChannelWisePooling('ksize',2,'stride',2),{'x13'},{name});
net.addLayer('cat1',dagnn.Concat('dim',3),{'conv3_reduced','conv4_reduced','conv5_reduced'},'cat1');
fea_dim =  (256+1)*256/2;
sprintf('channelwisefusion: fea_dim is %d',fea_dim)

net.addLayer('norm_for_cov',dagnn.ChannelWiseL2Norm(),{'cat1'},{'norm_for_cov','norm_for_cov_norm'},{})
new_layer_name = 'logm_cov_pool';
net.addLayer(new_layer_name , ...
                 dagnn.Logm_COV_Pool_C( 'epsilon', 5e-3), ...
                 'norm_for_cov', ...
                      {new_layer_name, [new_layer_name, '_aux_S'], [new_layer_name, '_aux_V']}) ;
                  
nonftscpnnDirr = ['nonftscpnn','/',opts.dataset_name];


if opts.initilizeFromLR      
    % get bcnn feature for train and val sets
    train = find(imdb.images.set==1|imdb.images.set==2);
     if ~exist(nonftscpnnDirr,'dir')
        mkdir(nonftscpnnDirr)
     end
     if numel(dir(nonftscpnnDirr))<numel(train)

    useGpu=1;
    batchSize= 64;
    prefetch = 1;
        if useGpu
            net.move('gpu') ;
        end

 
        imageSize = net.meta.normalization.imageSize;
        % compute and cache the bilinear cnn features
        for t=1:batchSize:numel(train)
            fprintf('Initialization: extracting bcnn feature of batch %d/%d\n', ceil(t/batchSize), ceil(numel(train)/batchSize));
            batch = train(t:min(numel(train), t+batchSize-1));
            imagePaths = imdb.images.name(batch);
            input = getImageBatch(imagePaths, 'subtractAverage',opts.averageImage,...
                                               'imageSize',imageSize(1:2));

            if useGpu
                input = gpuArray(input);
            end
            inputs = {'input',input};
            net.mode = 'test' ;
            net.eval(inputs);
            fIdx = net.getVarIndex('logm_cov_pool');
            code_b = net.vars(fIdx).value;
            code_b = squeeze(gather(code_b));

            for i=1:numel(batch)
                code = code_b(:,i);
                savefast(fullfile(nonftscpnnDirr, ['scpnn_nonft_', num2str(batch(i), '%05d')]), 'code');
            end
        end

        % move back to cpu
        if useGpu
            net.move('cpu');
        end
       end

 
        scal = 1;
        init_bias = 0;
        initialW = 0.01*scal*randn(1,1,fea_dim, opts.numClasses,'single');
        initialBias = init_bias.*ones(1, opts.numClasses, 'single');
        netc.layers = {};
        netc.layers{end+1} = struct('type', 'conv', 'name', 'classifier', ...
            'weights', {{initialW, initialBias}}, ...
            'stride', 1, ...
            'pad', 0, ...
            'learningRate', [1 2], ...
            'weightDecay', [1 0]) ;
        netc.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
        netc = vl_simplenn_tidy(netc) ;


        inittrain.batchSize = 64 ;
        inittrain.numSubBatches = 1;
        inittrain.numEpochs = 100 ;
        inittrain.continue = false ;
        inittrain.gpus = [1] ;
        inittrain.prefetch = false ;
        inittrain.learningRate = 0.001 ;
        inittrain.expDir = ['stage_one_train','/', opts.dataset_name] ;

        % sart: get the pretrain linear classifier
        if exist(fullfile(inittrain.expDir, 'initial_fc.mat'), 'file')
            load(fullfile(inittrain.expDir, 'initial_fc.mat'), 'netc') ;
        else

        bcnndb = imdb;
        tempStr = sprintf('%05d\t', train);
        tempStr = textscan(tempStr, '%s', 'delimiter', '\t');
        bcnndb.images.name = strcat('scpnn_nonft_', tempStr{1}');
        bcnndb.images.id = bcnndb.images.id(train);
        bcnndb.images.label = bcnndb.images.label(train);
        bcnndb.images.set = bcnndb.images.set(train);
        bcnndb.imageDir = nonftscpnnDirr;

        %train logistic regression
        sprintf('Strat training stage one...')
        [netc, info] = cnn_train(netc, bcnndb, @getBatch_bcnn_fromdisk, inittrain, ...
            'conserveMemory', true);
        save(fullfile(inittrain.expDir, 'initial_fc.mat'), 'netc', '-v7.3') ;
        sprintf('End training stage one!!!')
      end
 end

net.addLayer('classifier', dagnn.Conv('size', [1 1 fea_dim opts.numClasses], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), new_layer_name, {'classifier'},{'classifier_f'  'classifier_b'});
net.addLayer('prob', dagnn.SoftMax(), {'classifier'}, {'prob'}, {});
net.addLayer('objective', dagnn.Loss('loss', 'log'), {'prob', 'label'}, {'objective'}, {});
net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'prob','label'}, 'error') ;
% end
 
%-----------------------------------------------------------------------------
% start :initialize the last fully conenction layer with the weights from 'netc'

classiofier_layer_id = net.getLayerIndex('classifier');
f_ind = net.layers(classiofier_layer_id).paramIndexes(1);
b_ind = net.layers(classiofier_layer_id).paramIndexes(2);
if opts.initilizeFromLR
    classifier_w = netc.layers{1,1}.weights{1};
    classifier_b = netc.layers{1,1}.weights{2};
else
    classifier_w = init_weight(opts,[1 1 fea_dim opts.numClasses],'single');
    classifier_b = zeros([opts.numClasses, 1], 'single');
end

net.params(f_ind).value = classifier_w;
net.params(f_ind).learningRate = opts.newlayerlr;
net.params(f_ind).weightDecay = opts.newlayerdecay ;

net.params(b_ind).value = classifier_b;
net.params(b_ind).learningRate = 2*opts.newlayerlr;
net.params(b_ind).weightDecay = 0;
% end
%---------------------------------------------------------------------------------

for  i= 1:2:numel(net.params)-2
    net.params(i).learningRate = 1;
    net.params(i).weightDecay = 1;
    net.params(i+1).learningRate = 2;
    net.params(i+1).weightDecay = 0;
end

net.meta.inputSize = [net.meta.normalization.imageSize, 32] ;
net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 256 ; 
net.meta.normalization.averageImage = opts.averageImage ;
net.meta.classes.name = opts.classNames ;
net.meta.augmentation.jitterLocation = false ;
net.meta.augmentation.jitterFlip = true ;
%net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;
net.meta.augmentation.jitterBrightness = 0 ;
%net.meta.augmentation.jitterAspect = [2/3, 3/2] ;
net.meta.augmentation.jitterAspect = 0 ;
net.meta.trainOpts.learningRate = 0.001;
net.meta.trainOpts.numEpochs = 100 ;
net.meta.trainOpts.batchSize = 64 ; % number of batch
net.meta.trainOpts.weightDecay = 0.0005 ;
opts.expDir = ['stage_two_train','/',opts.dataset_name];
opts.train = struct() ;
opts.train.gpus = [1];
opts.numFetchThreads =4;
opts.networkType = 'dagnn';
sprintf('Start training stage two...')
[net, info] = cnn_train_dag(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      net.meta.trainOpts, ...
                      opts.train,'continue',false) ;
sprintf('End training stage two!!!')
end
