function [output, derivata] = identity(input)
%   IDENTITY Funzione di output dei nodi
%   Input:
%   "input" rappresenta l'input passato al nodo
%   Output:
%   "output" rappresenta l'output restituito dalla funzione
%   "derivata" rappresenta la derivata della funzione identit�

    output = input;

    derivata = 1;
end
