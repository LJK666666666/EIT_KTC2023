import scipy as sp
import main
import KTCScoring
import matplotlib.pyplot as plt
# import warnings


main.main('TrainingData', 'Output', 3)

score = 0
for ii in range(4):

    reco = sp.io.loadmat('Output/' + str(ii+1) + '.mat')
    reco = reco["reconstruction"]
    truth = sp.io.loadmat('GroundTruths/true' + str(ii+1) + '.mat')
    truth = truth["truth"]

    s = KTCScoring.scoringFunction(truth, reco)
    print('Score from target ' + str(ii+1) + ' = ' + str(s))
    score = score + s

print('Final score: ' + str(score) + ' / 4.00')
