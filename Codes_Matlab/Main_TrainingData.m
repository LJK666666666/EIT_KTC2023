%This script runs the main.m function for input file 'TrainingData', and
%then assesses all the output reconstructions using the given scoring
%algorithm.
close all; clear all; clc;

path(path,'MiscCodes/')
main('TrainingData', 'Output', 1);

score = 0;
for ii = 1:4

    load(['Output/' num2str(ii) '.mat']);
    load(['GroundTruths/true' num2str(ii) '.mat']);

    s = scoringFunction(truth, reconstruction);
    disp(['Score from target ' num2str(ii) ' = ' num2str(s)])
    score = score + s;

end

disp(['Final score: ' num2str(score/4) ])

