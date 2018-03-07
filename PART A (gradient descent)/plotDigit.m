function plotDigit(imageSet, labelSet, elem)
%   PLOTDIGIT Funzione che mostra l'immagine selezionata
%   Input:
%   "imageSet" rappresenta un insieme delle immagini 
%   "labelSet" rappresenta l'insieme dei target associati alle immagini
%   dell'imageSet
%   "elem" rappresenta l'indice dell'immagine presa in considerazione

    figure
    colormap(gray)  
    digit = imageSet(:,elem);           %   Prende l'immagine, rappresentata in colonna dall'insieme
    digit = reshape(digit, [28,28]);    %   Trasforma l'immagine in griglia 28x28
    digit = imresize(digit, [14,14]);   %   Dimezza le dimensioni dell'immagine
    imagesc(digit)
    title(strcat('Elem # ', num2str(elem), ', Actual digit: ',num2str(labelSet(elem))));
end
