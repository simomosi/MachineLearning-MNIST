function [crosserror, derivate] = crossEntropyError( outputs, targets )
%   CROSSENTROPYERROR Funzione di calcolo errore
%   Input: 
%   "outputs" rappresenta l'output della rete
%   "targets" rappresenta i target riferiti dell'immagine passata in input alla rete
%   Output:
%   "crosserror" rappresenta l'errore calcolato
%   "derivate" rappresenta la derivata dell'errore calcolata

%   Controllo utilizzato per evitare log(0)
     if sum(outputs == 0) > 0
         outputs = 0+eps;
     elseif sum(outputs == 1) > 0
         outputs = 1+eps;
     end

     if ((sum(log(outputs)) == -inf ) || (sum(log(1-outputs)) == -inf))
         outputs = outputs + eps;
     end
    
%   Calcolo errore tramite Cross Entropy
    crosserror = -(sum(targets .* log(outputs) + (1-targets) .* log(1-outputs))); 

%   Derivata della Cross Entropy
    derivate = (targets - 1)./(outputs - 1) - targets./outputs;
end



