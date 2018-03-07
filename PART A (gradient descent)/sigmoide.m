function [output, derivata] = sigmoide( z )
    %SIGMOIDE Funzione di output
    %   Output di tipo sigmoidale dei nodi interni.
    %   La funzione restituisce:
    %   [*] valori = 1/2 se l'input è 0;
    %   [*] valori = 1 se il limite dei valori dell'input tende a +inf;
    %   [*] valori = 0 se il limite dei valori dell'input tende a -inf;
    %Funzione sigmoide
    output = 1./(1 + exp(-z));
    %Derivata della funzione
    derivata = output .* (1 - output);

end


