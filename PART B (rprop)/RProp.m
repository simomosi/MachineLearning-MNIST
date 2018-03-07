function net = RProp(net)
%   GRADIENTDESCENT funzione di aggiornamento dei pesi
%   Input:
%   "net" rappresenta la rete
%   Output:
%   "net" rappresenta la rete aggiornata 
    DELTA_MIN = 1e-6;   % Limite inferiore del passo
    DELTA_MAX = 50;     % Limite superiore del passo
     
    for level=1:net.layers % Per ogni livello 
        temp_delta = net.deltaMatrix{level};      
        temp_input = net.activations{level};
        newDerivate =  temp_input * temp_delta;  
        product = newDerivate .* net.old.derivates{level}';  % Prodotto tra la derivata al tempo t e la derivata al tempo t-1
        if product > 0
            % Calcoliamo la variazione al tempo t prendendo il minimo tra la variazione al tempo t-1 ed il DELTA_MAX
            variazione = min(net.etaPlus * net.old.variazioni{level}, DELTA_MAX);
            % Aggiungiamo la variazione al peso il cui segno dipende dalla derivata
            net.weights{level} = net.weights{level} + (-sign(newDerivate).*variazione)'; 
        elseif product < 0
            % Calcoliamo la variazione al tempo t prendendo il massimo tra la variazione al tempo t-1 ed il DELTA_MIN
            variazione = max(net.etaMinus * net.old.variazioni{level}, DELTA_MIN);
            % Aggiungiamo la variazione al peso il cui segno dipende dalla derivata
            net.weights{level} = net.weights{level} - variazione'; 
            newDerivate = newDerivate * 0;
        else
            % Altrimenti prendiamo la variazione al tempo t-1
            variazione = net.old.variazioni{level};
            % Aggiungiamo la variazione al peso il cui segno dipende dalla derivata
            net.weights{level} = net.weights{level} + (-sign(newDerivate).*variazione)'; 
        end  
        net.old.variazioni{level} = variazione;     % Salviamo la variazione al tempo t
        net.old.derivates{level} = newDerivate';    % Salviamo la derivata al tempo t
    end  
    
    % Calcolo dei deltaBias aggiornati
    for level=1:net.layers % Per ogni livello 
        temp_delta = net.deltaBias{level};  
        temp_input = ones(1,length(temp_delta));
        newDerivate =  temp_input * temp_delta; 
        product = newDerivate .* net.old.deltaBias{level}';  % Prodotto tra il delta del bias tempo t ed il delta al tempo t-1
        if product > 0
            % Calcoliamo la variazione al tempo t prendendo il minimo tra la variazione al tempo t-1 ed il DELTA_MAX
            variazione = min(net.etaPlus * net.old.variazioniBias{level}, DELTA_MAX);
            % Aggiungiamo la variazione al peso il cui segno dipende dalla derivata
            net.biases{level} = net.biases{level} + (-sign(newDerivate).*variazione')'; 
        elseif product < 0
            % Calcoliamo la variazione al tempo t prendendo il massimo tra la variazione al tempo t-1 ed il DELTA_MIN
            variazione = max(net.etaMinus * net.old.variazioniBias{level}, DELTA_MIN);
            % Aggiungiamo la variazione al peso il cui segno dipende dalla derivata
            net.biases{level} = net.biases{level} - variazione; 
            newDerivate = newDerivate * 0;
        else
            % Altrimenti prendiamo la variazione al tempo t-1
            variazione = net.old.variazioniBias{level};
            % Aggiungiamo la variazione al peso il cui segno dipende dalla derivata
            net.biases{level} = net.biases{level} + (-sign(newDerivate).*variazione')'; 
        end  
        net.old.variazioniBias{level} = variazione;     % Salviamo la variazione al tempo t
        net.old.deltaBias{level} = newDerivate';
    end
   
end

