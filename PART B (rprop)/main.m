clear       %   Pulisce il prompt e dealloca tutto
close all
clc 

%   Importazione percorsi
thisFile = mfilename('fullpath');               % Percorso per 'main.m'
thisFolder = fileparts(thisFile);               % Percorso per 'PART-B' directory
projRoot = fileparts(fileparts(thisFolder));    % Percorso per la root directory 'Mlea'
addpath(thisFolder);
addpath(fullfile(projRoot, 'Data Set', 'loadMnist')); % Percorso per le funzioni loadMNIST

% Creazione della rete
rng('shuffle');         % Seed per i pesi random
sprintf('PARTE B (Resilient Backpropagation)')
net = createNetwork()

% Importazione insiemi 
images = loadMNISTImages(fullfile(projRoot, 'Data Set', 'Training Set', 'train-images.idx3-ubyte')); %784x60000, un'img per ogni colonna
labels = loadMNISTLabels(fullfile(projRoot, 'Data Set', 'Training Set', 'train-labels.idx1-ubyte')); %60000x1
sequence = randperm(60000);
 
% Creazione dei diversi Set di immagini
[ trainingSet, validationSet, testSet ] = initSets( net, sequence, labels ) ;

% Dichiarazione ed inizializzazione delle variabili usate
partialError = 0;               % Somma degli errori della singola epoca
partialDerivate = 0;            % Somma delle derivate degli della singola epoca
erroriEpoche = {};              % Contiene TUTTI gli errori PER OGNI immagine PER OGNI epoca
erroriDerivateEpoche = {};      % Contiene TUTTE le derivate PER OGNI immagine PER OGNI epoca
graficoErroriValidationSet = intmax;   % Diminuirï¿½ dopo la prima iterazione
graficoErroriValidationSet = [];

% Inserimento delle img con resize e reshape all'interno dell'insieme per il batch training
batchImgTraining = []; % Insieme di tutte le img del training set 
batchTargetTraining = []; % Insieme dei target delle img del training set 
for n = 1:net.trainingset
    chosenN = trainingSet(n); 
    chosenImg = formatImg(images, labels, chosenN);
    chosenLab = labels(chosenN); 
    batchImgTraining(:,n) = chosenImg;
    batchTargetTraining(:,n) = chosenLab == net.digit;
end

% Inserimento delle img con resize e reshape all'interno dell'insieme
batchImgValidation = []; % Insieme di tutte le img del validation set 
batchTargetValidation = []; % Insieme dei target delle img del validation set 
for n = 1:net.validationset
    chosenN = validationSet(n); 
    chosenImg = formatImg(images, labels, chosenN);
    chosenLab = labels(chosenN); 
    batchImgValidation(:,n) = chosenImg;
    batchTargetValidation(:,n) = chosenLab == net.digit;
end

% Inserimento delle img con resize e reshape all'interno dell'insieme
batchImgTest = []; % Insieme di tutte le img del test set 
batchTargetTest = []; % Insieme dei target delle img del test set 
for n = 1:net.testset
    chosenN = testSet(n); 
    chosenImg = formatImg(images, labels, chosenN);
    chosenLab = labels(chosenN); 
    batchImgTest(:,n) = chosenImg;
    batchTargetTest(:,n) = chosenLab == net.digit;
end

%START
erroreTrainingLast = 0;
for k = 1:net.epochs % Per ogni epoca
    % TRAINING
    [output, net.activations, net.derivates] = feedforward(batchImgTraining,net);
    [errore, derivata_errore] = net.err_function(output, batchTargetTraining); % Funzione di errore estratta dal vettore di function handles
    [net.deltaMatrix, net.deltaBias] = backpropagation(net, derivata_errore);
    net = RProp(net);
    erroreTraining = errore;
    graficoErroriTrainingSet(k) = sum(errore); % Array con gli errori per ogni epoca
     
    % VALIDATION
    erroreVS = 0; %     Errore sul validationSet
    [output, x, y] = feedforward(batchImgValidation,net);
    [errore, z] = net.err_function(output, batchTargetValidation); 
    erroreVS = sum(errore); 
    graficoErroriValidationSet(k) = erroreVS; % Array con gli errori per ogni validation
    sprintf('Epoca %d - Errore ValidationSet: %g', k,erroreVS)
    epocheEffettuate = k; 
    erroreTrainingLast = erroreTraining;
    
    if k>=net.minEpochs && erroreTraining < net.minError % Stop del training se raggiungo l'errore minimo (ma dopo almeno 100 epoche)
        sprintf('L\''errore è sceso sotto la soglia minima (%g < %g), fine del training',erroreTraining, net.minError)
        break; 
    end
end

% TEST
[output, x, y] = feedforward(batchImgTest,net);
recOutputs = output';
recLabels = batchTargetTest';
success = round(output) >= 1;
  
sprintf('- - - - Risultati - - - -')

stampaGrafici( graficoErroriTrainingSet, graficoErroriValidationSet ) % Grafico
arrayResult = [recLabels, success', ((recLabels ~= net.digit) + (success == 1)' - 1)];
testResult = sprintf('TP + FP: %g%%', 100*sum(success)/(net.testset)) % Tutti i digit accettati
FP = sum(arrayResult(:,3) == 1); % Digit erroneamente accettati
FN = sum(arrayResult(:,3) == -1); % Digit erroneamente rifiutati
TP_TN = sum(arrayResult(:,3) == 0); % Digit correttamente classificati
testFP = sprintf('FP: %g%%',FP/net.testset*100)
testFN = sprintf('FN: %g%%',FN/net.testset*100)
testCorrette = sprintf('Risposte corrette: %g%%',TP_TN/net.testset*100) 
