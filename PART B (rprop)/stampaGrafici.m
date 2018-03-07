function stampaGrafici(graficoErroriTrainingSet, graficoErroriValidationSet)
figure
plot(graficoErroriTrainingSet)
hold on
plot(graficoErroriValidationSet)
ylabel('Errore')
xlabel('Epoca')
legend('Errori Training Set', 'Errori Validation Set')
hold off
end

