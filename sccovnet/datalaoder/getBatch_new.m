function varargout = getBatch_new(opts, useGpu, networkType, imdb, batch,imageSize)
% -------------------------------------------------------------------------
images =  imdb.images.name(batch) ;
if ~isempty(batch) && imdb.images.set(batch(1)) == 1
  phase = 'train' ; 
  opts.train.imageSize = [imageSize,imageSize];
else
  phase = 'test' ;
  opts.test.imageSize = [imageSize,imageSize];
end

data = getImageBatch(images, opts.(phase), 'prefetch', nargout == 0) ;
if nargout > 0
  labels = imdb.images.label(batch) ;
  switch networkType
    case 'simplenn'
      varargout = {data, labels} ;
    case 'dagnn'
      varargout{1} = {'input', data, 'label', labels} ;
  end
end