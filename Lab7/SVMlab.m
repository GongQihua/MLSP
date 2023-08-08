function [SVMresults] = SVMlab(TrainingLDA,TestingLDA,ker,C)

pro_train = TrainingLDA';
pro_test = TestingLDA';

for i = 1:40
    train_label(9*(i-1)+1:9*i,1) = i;
end

for i = 1:40
    test_label(i,1) = i;    
end

t = templateSVM('KernelFunction',ker,'BoxConstraint',C);%svm
svm_model = fitcecoc(pro_train, train_label,'Learners',t);
result = predict(svm_model,pro_test);%get prediction
acc = 0;
for i = 1:40 %calculate accuracy
    if result(i) == test_label(i)
        acc = acc+1;
    end
end
SVMresults = acc/40;

end