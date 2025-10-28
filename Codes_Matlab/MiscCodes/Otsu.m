function [level,x] = Otsu(image,nvals,figno)
% OTSU automatic thresholding

[histogramCounts,x] = hist(image(:),nvals);
%figure(figno), clf, hist(image(:),256), hold on;

total = sum(histogramCounts); % total number of pixels in the image

top = 256;
sumB = 0;
wB = 0; %weight for class 1 (class probability)
maximum = 0.0;
sum1 = dot(0:top-1, histogramCounts);
for ii = 1:top
    wF = total - wB; %weight for class 2
    if wB > 0 && wF > 0
        mF = (sum1 - sumB) / wF;
        val = wB * wF * ((sumB / wB) - mF) * ((sumB / wB) - mF);
        if ( val >= maximum )
            level = ii;
            maximum = val;
        end
    end
    wB = wB + histogramCounts(ii);
    sumB = sumB + (ii-1) * histogramCounts(ii);
end
%plot(x(level)*ones(2,1),[0,max(histogramCounts)],'LineWidth',2,'Color','r')
%title('histogram of image pixels')
%set(gcf,'Units','normalized','OuterPosition',[0.3 0.2 0.3 0.4])
end