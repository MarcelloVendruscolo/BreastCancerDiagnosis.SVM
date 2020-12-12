% Read and make the medical data points accessible:
dataframe = readmatrix('wbdc.csv');

% Randomly distribute/separate the avaialble data in 87.5% learning - 12.5% testing datasets:
[nrows,ncols] = size(dataframe);
ratio = 0.875;
idx = randperm(nrows); % Perform permutation of the rows
learning_dataframe = dataframe(idx(1:round(ratio*nrows)),:);
testing_dataframe = dataframe(idx(round(ratio*nrows)+1:end),:);

% Create label matrix and features matrix for training purposes, respectively:
y_matrix_training = learning_dataframe(:,2);
x_matrix_training = learning_dataframe(:,3:end);

% Number of data points (l) and number of features (n), respectively:
l = size(x_matrix_training,1);
n = size(x_matrix_training,2);

% Array of C values to observe behaviour of classifier as this constant change values:
array_C = [0.0000000001 0.00001 0.001 1 100 1000 10000 1000000 1000000000000];

% Helper arrays to store performance measurements for each value of C:
array_accuracy = zeros(length(array_C),1);
array_sensitivity = zeros(length(array_C),1);
array_specificity = zeros(length(array_C),1);

% Iterate over the values of C and perform classification task minimise(1/2)*(x'Hx)+f'x:
for iterator = 1:length(array_C)
    C = array_C(iterator);
    H = diag([ones(1, n), zeros(1, l + 1)]);
    f = [zeros(1,n), 0, C * ones(1,l)]';
    
    % Constraints in the form Ax <= b:
    p = diag(y_matrix_training) * x_matrix_training;
    A = -[p y_matrix_training eye(l)];
    c = -ones(l,1);
    
    % Boundaries in the form lb <= x <= ub:
    lb = [-inf * ones(n+1,1); zeros(l,1)];
    
    options = optimoptions(@quadprog,'MaxIterations',500);
    z = quadprog(H,f,A,c,[],[],lb,[],[],options);
    w = z(1:n,:);
    b = z(n+1,:);
    eps = z(n+2:end,:);
    
    % Create true value and feature matrices for testing purposes, respectively:
    y_matrix_true = testing_dataframe(:,2);
    x_matrix_prediction = testing_dataframe(:,3:end);
    
    % Create matrix to store predictions:
    number_predictions = size(x_matrix_prediction,1);
    y_matrix_prediction = zeros(number_predictions,1);
    
    % Helper arrays to store construct a confusion matrix:
    condition_positive = 0;
    condition_negative = 0;
    true_positive = 0;
    true_negative = 0;
    false_positive = 0;
    false_negative = 0;

    for index = 1:number_predictions
        intermediate_matrix = w'*x_matrix_prediction(index,:)';
        y_matrix_prediction(index) = intermediate_matrix + b;
        if y_matrix_true(index) == 1
            condition_negative = condition_negative + 1;
            if y_matrix_prediction(index) >= 1
                true_negative = true_negative + 1;
            else
                false_positive = false_positive + 1;
                disp(index);
            end
        elseif y_matrix_true(index) == -1
            condition_positive = condition_positive + 1;
            if y_matrix_prediction(index) <= -1
                true_positive = true_positive + 1;
            else
                false_negative = false_negative + 1;
                disp(index);
            end
        end
    end
    
    array_accuracy(iterator) = (true_positive + true_negative)/number_predictions;
    array_sensitivity(iterator) = true_positive*100/condition_positive;
    array_specificity(iterator) = true_negative*100/condition_negative;
end

% Plot accuracy:
figure();
scatter(log(array_C(2:end)), array_accuracy(2:end), 'filled');
hold on;
plot(log(array_C(2:end)), array_accuracy(2:end));
hold off;
xlabel('Log C values');
ylabel('Accuracy');

% Plot sensitivity and specificity:
figure();
plot(log(array_C(2:end)), array_sensitivity(2:end));
hold on;
plot(log(array_C(2:end)), array_specificity(2:end));
hold off;
legend('Sensitivity','Specificity');
xlabel('C values');
ylabel('Percentage');