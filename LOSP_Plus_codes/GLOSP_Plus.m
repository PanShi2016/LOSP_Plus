function [] = GLOSP_Plus() 
% Global Spectral Method

graphPath = '../example/Amazon/graph';
communityPath = '../example/Amazon/community';

% load graph
graph = loadGraph(graphPath);

% load truth communities
comm = loadCommunities(communityPath);

% choose a community from truth communities randomly
commId = randi(length(comm));

% choose 3 nodes from selected community randomly
seedId = randperm(length(comm{commId}),3);
seed = comm{commId}(seedId);

% grab subgraph from each seed set
sample = SampleGraph(seed,graph,2);

% preprocessing, delete isolated nodes
subgraph = graph(sample,sample);
idx = find(sum(subgraph)==0);

if length(idx) > 0
    sample = setdiff(sample,sample(idx));
end

% compute eigenspace associated with two leading eigenvectors
subgraph = graph(sample,sample);
p = zeros(1,length(sample));
[~, ind] = intersect(sample,seed);
p(ind) = 1/length(ind);

opts.v0 = p';
Nrw = NormalizedGraph(subgraph);
[V,D] = eigs(Nrw,2,'lr',opts);
[~,id] = sort(diag(D),'descend');
V = V(:,id);
V = real(V);

% get sparse vector by linear programming
v = pos1norm(V,ind);

% bound detected community by truth community size
if length(sample) < length(comm{commId})
    detectedComm = sample;
else
    [~,I] = sort(v,'descend');
    detectedComm = sample(I(1:length(comm{commId})));
end

% compute F1 score
jointSet = intersect(detectedComm,comm{commId});
jointLen = length(jointSet);
F1 = 2*jointLen/(length(detectedComm)+length(comm{commId}));

% printing out result
fprintf('The detected community is')
disp(detectedComm')
fprintf('The F1 score between detected community and ground truth community is %.3f\n',F1)

% save out result
savePathandName = '../example/Amazon/output_GLOSP_Plus.txt';
dlmwrite(savePathandName,'The detected community is','delimiter','');
dlmwrite(savePathandName,detectedComm','-append','delimiter','\t','precision','%.0f');
dlmwrite(savePathandName,['The F1 score between detected community and ground truth community are ' num2str(F1,'%.3f')],'-append','delimiter','');

end
