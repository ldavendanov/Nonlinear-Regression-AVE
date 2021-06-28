function indices = SparseResampling(X,m)

Z = linkage(X','ward');
T = cluster(Z,'maxclust',m);
indices = false(1,size(X,2));
for i=1:m
    indices(find(T==i,1,'first')) = true;
end