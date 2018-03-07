function [res, attivazioni, derivate]  = feedforward(input,net)
%   FEEDFORWARD Funzione per la propagazione in avanti
%   Input:
%   "input" rappresenta il vettore di feature
%   "net" rappresenta la rete
%   Output:
%   "res" rappresenta il risultato della propagazione in avanti
%   "attivazioni" rappresenta il valore di attivazione per ogni nodo (usato
%   successivamente per l'aggiornamento dei pesi)
%   "derivate" rappresenta il valore della derivata per ogni nodo (usato
%   successivamente per l'aggiornamento dei pesi)

%   Creazione cell array 
    attivazioni = {};
    derivate = {};
    
%   Propagazione in avanti
    for j=1:net.layers % Per ogni livello j 
        attivazioni{j} = input;
%       Calcolo dell'output  dello strato j
        a = net.weights{j}*input; 
        size_biases = size(a);
        biases = repmat(net.biases{j},1,size_biases(2));
        a = a + biases;
        
        [input, derivata] = net.act_functions{j}(a);  % Funzione di output estratta a run-time dal vettore di function handles
        derivate{j} = derivata';
    end
%   L'ultimo output � stato salvato nella variabile "input"
    res = input; 
end
