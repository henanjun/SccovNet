function [out,Us,Ss] = vl_mylog_vc(X, varargin)
%Y = VL_MYLOG(X, DZDY)
%LogEig layer

gpuFlag = isa(X,'gpuArray');
X = double(gather(X));
[h,w,nChannel, nBatch] = size(X);
% calculate covariance mat
N = h*w;  one_mat = ones(N,1);
I_hat = (eye(N)-(one_mat*one_mat')./N)./(N-1);

X_vec = zeros(nChannel,h*w,nBatch);
for ix = 1 : nBatch
    X_vec(:,:,ix) = reshape(X(:,:,:,ix),[h*w,nChannel])';
end

if nargin < 4
    opts.epsilon = 1e-3;% default value;
    opts = vl_argparse(opts,varargin) ;
    Us = zeros(nChannel,nChannel,nBatch);
    Ss = zeros(nChannel,nChannel,nBatch);
    for ix = 1 : nBatch
        [Us(:,:,ix),Ss(:,:,ix),~] = svd(X_vec(:,:,ix)*I_hat*X_vec(:,:,ix)'+eye(nChannel)*opts.epsilon);
    end
    Y = zeros(nChannel,nChannel, nBatch);
    for ix = 1:nBatch
        Y(:,:,ix) = Us(:,:,ix)*diag(log(diag(Ss(:,:,ix))))*Us(:,:,ix)';
    end
    dim = (nChannel+1)*nChannel/2;
    Y = SPD2Euclidean(Y);
    Y = reshape(Y,1,1,dim,[]);
    if gpuFlag
        out = gpuArray(single(Y));
    else
        out = single(Y);
    end
else
    Us = varargin{1};
    Ss = varargin{2};
    dzdy = varargin{3};
    dzdx = zeros(h,w,nChannel, nBatch);
    aux = double(squeeze(gather(dzdy)));
    for ix = 1:nBatch
        U = Us(:,:,ix); S = Ss(:,:,ix);
        diagS = diag(S);
        ind =diagS >(nChannel*eps(max(diagS)));
        Dmin = (min(find(ind,1,'last'),nChannel));     
        S = S(:,ind); U = U(:,ind);
        dLdC = Euclidean2SPD(aux(:,ix)); dLdC = symmetric(dLdC);
        dLdU = 2*dLdC*U*diagLog(S,0);
        dLdS = diagInv(S)*(U'*dLdC*U);
        if sum(ind) == 1 % diag behaves badly when there is only 1d
            K = 1./(S(1)*ones(1,Dmin)-(S(1)*ones(1,Dmin))'); 
            K(eye(size(K,1))>0)=0;
        else
            K = 1./(diag(S)*ones(1,Dmin)-(diag(S)*ones(1,Dmin))');
            K(eye(size(K,1))>0)=0;
            K(find(isinf(K)==1))=0; 
        end
        if all(diagS==1)
            dzdy = zeros(nChannel,nChannel);
        else
            dzdy = U*(symmetric(K'.*(U'*dLdU))+dDiag(dLdS))*U';
        end
        dzdx(:,:,:,ix) =  reshape(I_hat*X_vec(:,:,ix)'*(dzdy+dzdy'),[h,w,nChannel]); %warning('no normalization');
         %dydx(:,:,:,ix) =  (dzdx+dzdx')*X_vec(:,:,ix)*I_hat; %warning('no normalization');
    end
    if gpuFlag
         out = gpuArray(single(dzdx));
    else
         out = single(dzdx);
    end
 end
end
