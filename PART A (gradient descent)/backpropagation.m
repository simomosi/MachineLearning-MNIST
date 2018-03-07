function [deltaMatrix,deltaBias] = backpropagation(net, derivata_errore)
%   BACKPROPAGATION Funzione calcolo delta
%	Input:
%   "net" � la rete neurale
%   "derivata_errore" rappresenta la derivata dell'errore rispetto all'output
%	Output:
%	"deltaMatrix" � un cell array dove ogni cella contiene l'array dei valori delta per ogni strato

%   Creazione cell array
    deltaMatrix = {};
    deltaBias = {};
%   Calcolo dei delta
    for i=net.layers:-1:1 % Per ogni strato di nodi i
        if (i==net.layers)% Strato di output
            deltaMatrix{i} = net.derivates{i} .* derivata_errore'; 
        else % Strato di nodi interni
            deltaMatrix{i} = net.derivates{i} .* (deltaMatrix{i+1}*net.weights{i+1});
        end 
    end
    
    for i=net.layers:-1:1 % Per ogni strato di nodi i
        if (i==net.layers)% Strato di output
            deltaBias{i} = net.derivates{i} .* derivata_errore'; 
        else % Strato di nodi interni
            deltaBias{i} = net.derivates{i} .* (deltaBias{i+1}*net.biases{i+1});
        end 
    end 
end
