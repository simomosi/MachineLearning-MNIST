function [ trainingSet, validationSet, testSet ] = initSets( net, sequence, labels ) 
%   INITSETS Funzione di inizializzazione dei diversi insiemi di indici delle immagini
%   Input:
%   "net" rappresenta la rete
%   "sequence" rappresenta una permutazione di indici da 1 a 60000 senza ripetizioni
%   "labels" rappresenta l'insieme dei target
%   Output:
%   "trainingSet" rappresenta l'insieme di indici delle immagini usate come Training Set
%   "validationSet" rappresenta l'insieme di indici delle immagini usate come Validation Set
%   "testSet" rappresenta l'insieme di indici delle immagini usate come Test Set

%   Dichiarazione array
    tr_0 = []; % training set di immagini DIVERSE dal digit scelto
    tr_1 = []; % training set di immagini UGUALI al digit scelto
    vs_0 = []; % validation set di immagini DIVERSE dal digit scelto
    vs_1 = []; % validation set di immagini UGUALI al digit scelto
    te_0 = []; % test set di immagini DIVERSE dal digit scelto
    te_1 = []; % test set di immagini UGUALI al digit scelto
    
    for i=1:length(sequence) 
%       Il ciclo termina non appena sono stati riempiti tutti gli array
        if length(tr_0) >= net.trainingset/2 && length(tr_1) >= net.trainingset/2 && length(vs_0) >= net.validationset/2 && length(vs_1) >= net.validationset/2 && length(te_0) >= net.testset*0.90 && length(te_1) >= net.testset*0.10 
           break 
        end 
%       Scelgo l'immagine e il target associati all'indice dalla sequenza 
        chosenN = sequence(i);
        chosenLab = labels(chosenN);
   
%       In base al digit inserisco l'indice nell'insieme corrispondente (se non "pieno")
        if isTarget(chosenLab, net.digit) == 1 && length(tr_1) < net.trainingset/2
            tr_1(length(tr_1)+1) = chosenN;  
            
        elseif isTarget(chosenLab, net.digit) == 0 && length(tr_0) < net.trainingset/2
            tr_0(length(tr_0)+1) = chosenN;  
            
        elseif isTarget(chosenLab, net.digit) == 1 && length(vs_1) < net.validationset/2
            vs_1(length(vs_1)+1) = chosenN;  
            
        elseif isTarget(chosenLab, net.digit) == 0 && length(vs_0) < net.validationset/2
            vs_0(length(vs_0)+1) = chosenN;  
            
        elseif isTarget(chosenLab, net.digit) == 1 && length(te_1) < net.testset * 0.10
            te_1(length(te_1)+1) = chosenN; 
            
        elseif isTarget(chosenLab, net.digit) == 0 && length(te_0) < net.testset * 0.90
            te_0(length(te_0)+1) = chosenN; 
        end
    end
    
%   Unisco gli array creati e ne faccio una permutazione per ogni Set in
%   modo da avere una sequenza mista di target 0 e target 1
    trainingSet = [tr_0, tr_1];
    trainingSet = trainingSet(randperm(net.trainingset));    
    
    validationSet = [vs_0, vs_1];
    validationSet = validationSet(randperm(net.validationset));
    
    testSet = [te_0, te_1];
    testSet = testSet(randperm(net.testset));
end

