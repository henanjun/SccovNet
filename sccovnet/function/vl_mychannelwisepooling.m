function [y] = vl_mychannelwisepooling(x,dzdy, varargin)


[h,w,numChannels,numBatch]=size(x);

x = reshape(x,h*w,numChannels,1,numBatch);
opts.stride = 2;
opts.ksize = 2;
opts.pad = [0,0,0,0];
opts = vl_argparse(opts,varargin);
stride = opts.stride;
ksize = opts.ksize;
pad = opts.pad;
if isempty(dzdy)
    y = vl_nnpool(x,[1,ksize],'stride',[1 stride],'method','avg','CuDNN');
    y = reshape(y,h,w,[],numBatch);
else
    dzdy = reshape(dzdy,h*w,[],1,numBatch);
    y = vl_nnpool(x,[1,ksize],dzdy,'stride',[1 stride],'method','avg','CuDNN');
    y = reshape(y,h,w,[],numBatch);
end

end
