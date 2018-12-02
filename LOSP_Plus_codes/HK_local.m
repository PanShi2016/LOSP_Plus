function [] = HK_local() 
% Heat Kernel Based community Detection

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

% use heat kernel vector
hkvec = hkvec_mex(graph,seed,20,1e-4);
inds = hkvec(:,1);
vals = hkvec(:,2);

[~,I] = sort(vals,'descend');
set = inds(I);
set = union(seed,set,'stable');

% bound detected community by local minimal conductance
% compute conductance
subgraph = graph(set,set);
conductance = zeros(1,length(set));
for i = 1 : length(set)
    conductance(i) = getConductance(subgraph,[1:i]);
end

% compute first local minimal conductance
startId = 3;
index = GetLocalCond(conductance,startId,1.02);
detectedComm = set(1:index);

% compute F1 score
jointSet = intersect(detectedComm,comm{commId});
jointLen = length(jointSet);
F1 = 2*jointLen/(length(detectedComm)+length(comm{commId}));

% printing out result
fprintf('The detected community is')
disp(detectedComm')
fprintf('The F1 score between detected community and ground truth community is %.3f\n',F1)

% save out result
savePathandName = '../example/Amazon/output_HK_local.txt';
dlmwrite(savePathandName,'The detected community is','delimiter','');
dlmwrite(savePathandName,detectedComm','-append','delimiter','\t','precision','%.0f');
dlmwrite(savePathandName,'The F1 score between detected community and ground truth community is','-append','delimiter','');
dlmwrite(savePathandName,F1,'-append','delimiter','\t','precision','%.3f');

end
