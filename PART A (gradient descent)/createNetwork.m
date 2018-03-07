function neuralNet = createNetwork() 
%   CREATENETWORK Funzione creazione della rete
%   Output:
%   "neuralNet" � uno struct array che rappresenta la rete

    neuralNet.digit = 1;                                %   Numero da riconoscere 
    neuralNet.trainingset = 400;                      	%   Elementi del Training Set    
    neuralNet.validationset = ceil(neuralNet.trainingset/1);  %   Elementi del Validation Set
    neuralNet.testset = ceil(neuralNet.trainingset/2);        %   Elementi del Test Set
	neuralNet.epochs = 100;                            %   Numero di Epoche
    neuralNet.minEpochs = max([50,ceil(neuralNet.epochs*0.25)]);        %   Numero minimo di Epoche da effettuare (almeno 50)
    neuralNet.dimension = 784/4;                        %   Dimensione dell'input [immagini 14x14]
    neuralNet.weights = {};                             %   Cell array delle matrici dei pesi per ogni strato di nodi
    neuralNet.biases = {};                              %   Cell array dei vettori dei bias per ogni strato di nodi
    neuralNet.deltaMatrix = {};                         %   Cell array delle matrici dei delta per ogni strato di nodi
    neuralNet.deltaBias = {};
    neuralNet.act_functions = {};                       %   Cell array delle funzioni di attivazione per ogni strato di nodi
%   neuralNet.act_functions = {@sigmoide, @identity};   %   Esempio (specificare una funzione per ogni strato)
    neuralNet.err_function = {};                        %   Cell array della funzione di errore [Somma dei quadrati \ CrossEntropy]
%   neuralNet.err_function = {@squareSum};              %   Esempio
    neuralNet.eta = 0.4;                                %   Valore di eta 
    neuralNet.nodes4layer = [3, 1];                    %   Vettore dei nodi per ogni strato
    neuralNet.layers = length(neuralNet.nodes4layer);   %   Numero di strati della rete
    neuralNet.activations = {};                         %   Cell array delle attivazioni al tempo t-1 per ogni strato di nodi
    neuralNet.derivates = {};                           %   Cell array delle derivate per ogni strato di nodi
    
%   Il valore massimo per i vari Set � impostato intorno ad 8000 per mancanza di cifre
    if (neuralNet.trainingset > 8000)
        neuralNet.trainingset = 8000;   
        neuralNet.validationset = neuralNet.trainingset/4;
        neuralNet.testset = neuralNet.trainingset/4;
    end
%   Inizializzazione di alcune matrici per ogni strato di nodi
    for i=1:length(neuralNet.nodes4layer) 
        if i > 1 % Strato nodi hidden
            neuralNet.weights{i} = rand(neuralNet.nodes4layer(i), neuralNet.nodes4layer(i-1));
        else % Strato nodi input 
            neuralNet.weights{i} = rand(neuralNet.nodes4layer(i), neuralNet.dimension); 
        end
        neuralNet.biases{i} = rand(neuralNet.nodes4layer(i), 1); 
    end
    
%   Inizializzazioni funzioni di output dei diversi strati
%   Funziona solo se NON � stata specificata una funzione di attivazione
%   per ogni livello della rete (o se � stato inserito un errato numero di
%   puntatori a funzione)
    if length(neuralNet.act_functions) ~= length(neuralNet.nodes4layer)
%       Funzioni implementate: @sigmoide  @identity
        hidden_function = @sigmoide;
        output_function = @identity;
        
        neuralNet.nlayers = length(neuralNet.nodes4layer); 
%       Assegna le funzioni scelte a seconda degli strati disponibili 
        for i=1:length(neuralNet.nodes4layer)-1
            neuralNet.act_functions{i} = hidden_function;
        end
        neuralNet.act_functions{length(neuralNet.nodes4layer)} = output_function;  
    end  
    
%   Inizializzazione funzione di errore
%   Funzioni implementate: @squareSum  @crossEntropyError
    neuralNet.err_function = @squareSum;

%   Controlli parametri inseriti
    if (length(neuralNet.nodes4layer) < 1) %    Almeno uno strato di output
        exit(-1);
    end

    for i=1:length(neuralNet.nodes4layer)
        nodes = neuralNet.nodes4layer(i);
        if (nodes < 1)                     %   Almeno un nodo per strato
            exit(-1);
        end
    end
end

