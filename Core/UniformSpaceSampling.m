function ind = UniformSpaceSampling(X,m,Method)

N = size(X,2);

if nargin < 3
    Method = 'Clustering';
end

switch Method
    
    case 'Distance'
        
        ind = zeros(1,m);
        ind(1) = randi([1 N]);
        
        for j=2:m
            
            d = zeros(j-1,N);
            for i=1:j-1
                e = repmat( X(:,ind(i)), 1, N ) - X;
                d(i,:) = diag(e'*e);
            end
            
            d = prod(d,1);
            d(ind(1:j-1)) = 0;
            
            [~,k] = max(d);
            ind(j) = k;
            
        end
        
    case 'Clustering'
        
        Z = linkage(X','ward');
        T = cluster(Z,'maxclust',m);
        
        indx = 1:N;
        ind = zeros(1,m);
        
        for i=1:m
            ind0 = indx(T == i);
            ind1 = randi(numel(ind0));
            ind(i) = ind0(ind1);
        end
        
end