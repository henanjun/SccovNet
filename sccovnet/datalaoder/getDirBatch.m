function inputs = getDirBatch( opts,allsamples_name, idex_SL,sz,batch,train_average)
train_id = idex_SL(1,:);
train_label = idex_SL(2,:);
for i = 1:length(train_id)
    trainsample_name{i} = allsamples_name{train_id(i)};
end

for j = 1:length(batch)
    trainsample_batch_name{j} = trainsample_name{batch(j)};
end
tmp =  vl_imreadjpeg(trainsample_batch_name, 'NumThreads', 5,'Resize',sz);
for i = 1:length(batch)
    images(:,:,:,i) = tmp{i}-train_average;
end
labels = train_label(batch);

if  opts.useGpu > 0
      images = gpuArray(images);
end
inputs = {'input',images,'label',labels};

end