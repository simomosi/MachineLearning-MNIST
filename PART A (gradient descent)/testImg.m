% Funzione di test di una singola immagine
% Stampa il digit da riconoscere, l'etichetta dell'immagine numero chosenN,
% l'output della feed forward e l'errore calcolato
% INDICE_IMMAGINE da 1 a length(images)
function testImg( chosenN , images, labels, net)
        chosenImg = images(:,chosenN);
        chosenImg = reshape(chosenImg, [28,28]); 
        chosenImg = imresize(chosenImg, [14,14]);  
        chosenImg = reshape(chosenImg, [196,1]); 
        chosenLab = labels(chosenN);

        [output, attivazioni, derivate] = feedforward(chosenImg,net);
        [errore, derivata_errore] = net.err_function(output, isTarget(chosenLab, net.digit)); 
        sprintf('Digit: %d - Etichetta: %d - Output: %g - Errore: %g',net.digit,chosenLab,output,errore) 
end

