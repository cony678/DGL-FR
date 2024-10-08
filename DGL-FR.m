function [ W, obj ] = DGL-FR( X, Y, para )

[num_train, num_feature] = size(X); num_label = size(Y, 2);

% calculate X graph Laplacian
options1 = [];
options1.NeighborMode = 'KNN';
options1.k = 0;
options1.WeightMode = 'HeatKernel';
options1.t = 1;
S_x = constructW(X,options1);
D_x = diag(sum(S_x,2));
L_x = D_x - S_x;

% Initialize W
W = rand(num_feature, num_label);
% Initialize Z
Z = rand(num_train, num_label);
%define A
A = diag(10000000+zeros(1,num_train)); %2147483647
I = ones(num_label,1);


FF = zeros(num_feature,num_feature);
for i=1:num_feature     %mutual information
    for j=1:num_feature
        % The redundancy betwwen features
        FF(i,j)=mi(X(:,i),X(:,j));
    end
end
for i=1:num_feature
    FF(i,i)=1;
end

iter = 1; objp = 1; 

while 1
    
    % Update U
    u = 0.5./sqrt(sum(W.*W, 2) + eps);
    U = diag(u);
   
    % Update L_z  
    options1 = [];
    options1.NeighborMode = 'KNN';
    options1.k = 0;
    options1.WeightMode = 'HeatKernel';%'Cosin';
    options1.t = 1;
    S_z = constructW(Z',options1);
    D_z = diag(sum(S_z,2));
    L_z = D_z - S_z;
   
    % Update W
    W_1 = W .* ((X'*Z + para.beta*W*S_z)./((X'*X*W + para.beta*W*D_z + para.gamma*U*W +para.lamda*FF*W)+eps));
    %Update T
    R = - S_z .* (diag(W'*W)*I' + I*diag(W'*W)'-2*W'*W);
    D_R = diag(sum(R,2));
    T = D_R-R;
    % Update Z
    Z_1 = Z .*((X*W + para.alpha*S_x*Z + para.alpha*A*Y - para.beta*Z*D_R)./((Z + para.alpha*D_x*Z + para.alpha*A*Z - para.beta*Z*R)+eps));
     
    W = W_1;
    Z = Z_1;
    % objective function
    a = trace(Z'*L_x*Z);
    b = trace((Z-Y)'*A*(Z-Y));
    c = (norm((X*W - Z), 'fro'))^2;
    d = trace(W*L_z*W');
    e = sum(sqrt(sum(W.*W,2)+eps));
    f = trace(W'*FF*W);  %mutual information
    %obj(iter) =  para.alpha * (trace(Z'*L_x*Z) + trace((Z-Y)'*A*(Z-Y))) + (norm((X*W - Z), 'fro'))^2  + para.beta * trace(W*L_z*W') + para.gamma * sum(sqrt(sum(W.*W,2)+eps));
    obj(iter) =  para.alpha * (a + b) + c + para.beta * d + para.gamma * e  + para.lamda*f;%
    %obj(iter) =  para.alpha * (a ) + c + para.beta * d + para.gamma * e  + para.lamda*f;
    
    cver = abs((obj(iter) - objp)/objp);
    objp = obj(iter);
    iter = iter + 1;
    disp(iter);

    % convergence condition
    if (cver < 10^-3 && iter >2) , break, end
    
end

end

