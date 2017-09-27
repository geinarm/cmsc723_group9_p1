
%% Generate points that are rougly co-planar
v1 = rand(1,3) - 0.5;
v2 = [v1(1), -v1(2), v1(3)];
%v2 = rand(1,3) - 0.5;

v1 = v1./norm(v1);
v2 = v2./norm(v2);

V1 = repmat(v1, [50, 1]);
V2 = repmat(v2, [50, 1]);

P = V1 .* rand(50,1) + V2 .* rand(50,1);
P = P + (rand(50, 3) .* 0.1);

%plot3(P(:, 1), P(:, 2), P(:, 3), '.', 'MarkerSize', 5);
%hold on;


X = P(1:30, :);
mu = mean(X);
X = X - mu;

plot3(X(:, 1), X(:, 2), X(:, 3), '.', 'MarkerSize', 10, 'color', 'b');
hold on;
xlim([-1, 1]);
ylim([-1, 1]);
zlim([-1, 1]);

[U, S, V] = svd(X, 'econ');

X_ = (V' * X')'; 
X_(:, 3) = 0;

plot3(X_(:, 1), X_(:, 2), X_(:, 3), 'o', 'MarkerSize', 10);

P_ = P - mu;
P_ = (V' * P_')';
%P_(:, 3) = 0;
plot3(P_(:, 1), P_(:, 2), P_(:, 3), '.', 'MarkerSize', 15, 'color', 'r');