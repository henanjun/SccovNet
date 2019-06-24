function outPoints = SPD2Euclidean(inPoints)
[m, nFeatures, nPoints] = size(inPoints);
tmpSPD = ones(nFeatures);
tmpSPD(tril(tmpSPD) == 1) = 0;
tmpIdx = tmpSPD > 0;
nFeatures = ((2*nFeatures+1)^2-1)/8;
outPoints = zeros(nFeatures,nPoints);

for tmpC1 = 1:nPoints
    tmp = inPoints(:,:,tmpC1);
    tmpSPD = zeros(nFeatures,1);
    tmpSPD(1:m,:) = diag(tmp);
    tmpSPD(m+1:nFeatures) = tmp(tmpIdx).*sqrt(2);
    outPoints(:,tmpC1) = tmpSPD;    
end %end tmpC1
