function [weights,biases] = gradientDescent(net) 
%   GRADIENTDESCENT funzione di aggiornamento dei pesi
%   Input:
%   "net" rappresenta la rete
%   Output:
%   "weights" rappresenta il cell array con le matrici dei pesi per
%   ciascuno strato della rete

    for level=1:net.layers % Per ogni strato di nodi prende l'input e i delta
        temp_input = net.activations{level};
        temp_delta = net.deltaMatrix{level};
%       Calcola la variazione dei pesi e aggiorna lo strato di pesi attuale (come prodotto tra matrici)
        variazione = -net.eta * (temp_input * temp_delta);
        weights{level} = net.weights{level} + variazione';
        
        temp_delta = net.deltaBias{level};
        biases{level} = net.biases{level} + (-net.eta * temp_delta');
    end 
end
