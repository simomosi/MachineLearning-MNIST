function loadImgPaint( net )
    chosenImg = images(:,chosenN);
    chosenImg = reshape(chosenImg, [28,28]); 
    chosenImg = imresize(chosenImg, [14,14]);  
    chosenImg = reshape(chosenImg, [196,1]); 
    chosenLab = labels(chosenN);

    [output, attivazioni, derivate] = feedforward(chosenImg,net);
    [errore, derivata_errore] = net.err_function(output, isTarget(chosenLab, net.digit)); 
    "Digit: "+net.digit+" - Etichetta: "+chosenLab+" - Output: "+output+" - Errore: "+errore
end

