function [squareerror, derivata] = squareSum(outputs, targets)
%   SQUARESUM Funzione di errore
%   Input: 
%   "outputs" rappresenta l'output della rete
%   "targets" rappresenta i target dei singoli valori 
%   Output:
%   "squareerror" rappresenta l'errore calcolato
%   "derivate" rappresenta la derivata dell'errore calcolata

%   Calcolo errore tramite Somma dei quadrati
    squareerror = (sum(outputs - targets)^2) /2;
    
%   Derivata della Somma dei quadrati
    derivata = sum(outputs - targets);

end

