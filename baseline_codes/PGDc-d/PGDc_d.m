function [] = PGDc_d() 
% PGDc-d: The projected gradient descent algorithm for optimizing sigma-conductance

graphPath = '../../example/Amazon/graph';
communityPath = '../../example/Amazon/community';

% load graph
graph = loadGraph(graphPath);

% load truth communities
comm = loadCommunities(communityPath);

% choose a community from truth communities randomly
commId = randi(length(comm));

% choose 3 nodes from selected community randomly
seedId = randperm(length(comm{commId}),3);
seed = comm{commId}(seedId);

seed_vec = zeros(length(graph),1);
seed_vec(seed) = 1;

[set,l] = optimize_cluster(graph,seed_vec,'score','sigma');
set = find(set);

% compute F1 score and Jaccard index
jointSet = intersect(set,comm{commId});
jointLen = length(jointSet);

F1 = 2*jointLen/(length(set)+length(comm{commId})); 

% printing out result
fprintf('The detected community is')
disp(set')
fprintf('The F1 score between detected community and ground truth community are %.3f\n',F1)

% save out result
savePathandName = '../../example/Amazon/output_PGDc-d.txt';
dlmwrite(savePathandName,'The detected community is','delimiter','');
dlmwrite(savePathandName,set','-append','delimiter','\t','precision','%.0f');
dlmwrite(savePathandName,['The F1 score between detected community and ground truth community are ' num2str(F1,'%.3f')],'-append','delimiter','');

end
