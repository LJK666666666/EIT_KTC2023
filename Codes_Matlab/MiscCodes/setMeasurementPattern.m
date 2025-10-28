function [Inj,Mpat,vincl] = setMeasurementPattern(Nel)

Inj = toeplitz([1 -1 zeros(1,Nel-2)],[1 zeros(1,Nel-1)]);
Inj(1,end) = -1;
Mpat = Inj(:,1:Nel-1);
vincl = true(Nel*(Nel-1),1);

Inj = eye(Nel);

end