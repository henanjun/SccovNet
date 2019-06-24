function [y,tmp_norm] = vl_nnmychannelnorm(x,varargin)



gpuMode = isa(x, 'gpuArray');
[h, w, ch, bs] = size(x);


if nargin <2
    if gpuMode
      tmp_norm = gpuArray(zeros([h, w, ch, bs], 'single'));
    else
      tmp_norm = zeros([h, w, ch, bs], 'single');
    end
    for i = 1:bs
        tmp = reshape(x(:,:,:,i),h*w,ch);
        tmp_norm(:,:,:,i) = reshape(ones(h*w,1)*sqrt(sum(tmp.^2)),[h,w,ch])+1e-12;
    end
    y = x./tmp_norm;
else
    dzdy = varargin{2};
    tmp_norm = varargin{1};
    E = dzdy./tmp_norm;
    F = sum(sum(x.*dzdy));
    F = F.*tmp_norm.^(-3);
    F = bsxfun(@times, x, F);
    y = E-F;
end

end