clear 
close all
clc 
%   Importazione percorsi
thisFile = mfilename('fullpath');               % Percorso per 'main.m'
thisFolder = fileparts(thisFile);               % Percorso per 'PART-A' directory
projRoot = fileparts(fileparts(thisFolder));    % Percorso per la root directory 'Mlea'
addpath(thisFolder);
addpath(fullfile(projRoot, 'Data Set', 'loadMnist')); % Percorso per le funzioni loadMNIST

% Creazione della rete
rng('shuffle');         % Seed per i pesi random
sprintf('PARTE A (Discesa del gradiente)')
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
erroreValidationSet = intmax;   % Diminuir� dopo la prima iterazione
graficoErroriValidationSet = [];


%START

for k = 1:net.epochs % Per ogni epoca
   % TRAINING
    for n = 1:net.trainingset           % Per ogni immagine del training set
        chosenN = trainingSet(n);       % Scelgo l'n-esimo indice random
        chosenImg = formatImg(images, labels, chosenN);     % Resize dell'immagine
        chosenLab = labels(chosenN);    % Seleziono l'etichetta associata
        
        [output, net.activations, net.derivates] = feedforward(chosenImg,net);
        [errore, derivata_errore] = net.err_function(output, isTarget(chosenLab, net.digit)); % Funzione di errore estratta dal vettore di function handles
        
        [net.deltaMatrix, net.deltaBias] = backpropagation(net, derivata_errore);
        [net.weights,net.biases] = gradientDescent(net);
        
        erroriEpoche{k}(n) = errore; % Cell Array con l'errore per ogni epoca k di ogni immagine n 
        erroriDerivateEpoche{k}(n) = derivata_errore; % Cell Array con la derivata dell'errore per ogni epoca k di ogni immagine n
        
    end
    partialError = sum(erroriEpoche{k});
    partialDerivate = sum(erroriDerivateEpoche{k});
    graficoErroriTrainingSet(k) = partialError;
    
    %sprintf('Epoca %d - Errore training: %g', k,partialError)
    
    %VALIDATION
    erroreVS = 0; % Errore sul validation set
    for n = 1:net.validationset 
        chosenN = validationSet(n); 
        chosenImg = formatImg(images, labels, chosenN);
        chosenLab = labels(chosenN);

        [output, tmp_a, tmp_d] = feedforward(chosenImg,net);
        [errore, tmp_der] = net.err_function(output, isTarget(chosenLab, net.digit)); 
        erroreVS = erroreVS + errore;
    end
        
    graficoErroriValidationSet(k) = erroreVS;
    erroreValidationSet = erroreVS;
    
    sprintf('Epoca %d - Errore training: %g - Errore ValidationSet: %g', k,partialError,erroreVS)
    if k > net.minEpochs && graficoErroriValidationSet(k) > graficoErroriValidationSet(k-1)
        sprintf('L\''errore sul VS sta aumentando ((k)%g > (k-1)%g), fine del training', graficoErroriValidationSet(k), graficoErroriValidationSet(k-1))
        break;
    end
end

% TEST
success = [];
errors = 0;
recOutputs = [];
recLabels = [];
for n = 1:net.testset 
    chosenN = testSet(n); 
    chosenImg = formatImg(images, labels, chosenN);
    chosenLab = labels(chosenN);

    [output, tmp_a, tmp_d] = feedforward(chosenImg,net); 
    recOutputs(n) = output;
    recLabels(n) = chosenLab;
    success(n) = round(output) >= 1;
end

sprintf('- - - - Risultati - - - -')
stampaGrafici( graficoErroriTrainingSet, graficoErroriValidationSet ) % Grafico

arrayResult = [recLabels; success; ((recLabels ~= net.digit) + (success == 1) - 1)]';
testResult = sprintf('TP + FP: %g%%', 100*sum(success)/(net.testset)) % Tutti i digit accettati
FP = sum(arrayResult(:,3) == 1); % Digit erroneamente accettati
FN = sum(arrayResult(:,3) == -1); % Digit erroneamente rifiutati
TP_TN = sum(arrayResult(:,3) == 0); % Digit correttamente classificati
testFP = sprintf('FP: %g%%',FP/net.testset*100)
testFN = sprintf('FN: %g%%',FN/net.testset*100)
testCorrette = sprintf('Risposte corrette: %g%%',TP_TN/net.testset*100) 
