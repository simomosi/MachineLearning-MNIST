function result = isTarget(x, digit)
%   ISTARGET Funzione controllo target
%   Input:
%   "x" rappresenta il target da controllare
%   "digit" rappresenta l'elemento da riconoscere

    if x==digit
        result = 1;
    else 
        result = 0;
    end
end
