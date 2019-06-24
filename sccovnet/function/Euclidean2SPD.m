function outPoints = Euclidean2SPD(inPoints)
[nFeatures, nPoints] = size(inPoints);
nFeatures = (sqrt(1+8*nFeatures)-1)/2;
outPoints = zeros(nFeatures,nFeatures,nPoints);

tmpIdx1 = eye(nFeatures)==1;
tmpIdx2 = tril(ones(nFeatures))==0;

for tmpC1 = 1:nPoints
    tmp = inPoints(:,tmpC1);
    tmpSPD = zeros(nFeatures,nFeatures);
    tmpSPD(tmpIdx2) = tmp(nFeatures+1:end)./(sqrt(2));
    tmpSPD = tmpSPD+tmpSPD';
    tmpSPD(tmpIdx1) = tmp(1:nFeatures);
    outPoints(:,:,tmpC1) = tmpSPD;    
end %end tmpC1
