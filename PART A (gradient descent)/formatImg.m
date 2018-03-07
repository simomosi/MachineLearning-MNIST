function img = formatImg( imageSet, labelSet, elem )
%   FORMATIMG funzione di modifica della dimensione dell'immagine (reshape e resize)
%   Input:
%   "imageSet" rappresenta un insieme di immagini da cui prelevare la numero 'elem' 
%   "labelSet" rappresenta l'insieme dei target associati alle immagini dell'imageSet
%   "elem" rappresenta l'indice dell'immagine presa in considerazione
%   Output:
%   "img" rappresenta l'immagine modificata

    chosenImg = imageSet(:,elem);               %   Prende l'immagine, rappresentata in colonna
    chosenImg = reshape(chosenImg, [28,28]);    %   Trasforma l'immagine in griglia 28x28
    chosenImg = imresize(chosenImg, [14,14]);   %   Dimezza le dimensioni dell'immagine
    chosenImg = reshape(chosenImg, [196,1]);    %   Trasforma l'immagine in colonna
    
    img = chosenImg;
end

