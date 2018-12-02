function I2 = sampleGraph(seeds,graph,steps)
% Sample the subgraph according to BFS

for i = 1 : length(seeds)
    % one round BFS
    tempI = find(graph(seeds(i),:) > 0);
    I = union(seeds(i),tempI,'stable');
    newI = I;
    iter = 1;

    while (length(I) < 300 && iter <= steps)
        % filter some nodes
        degree = sum(graph);
        subdegree = degree(newI);
        subgraph = graph(newI,newI);
        subdegreeIn = sum(subgraph);

        degreeOut = zeros(1,length(newI));
        degreeOut = subdegree - subdegreeIn;

        inward_ratio = subdegreeIn./degreeOut;
        [~,ind] = sort(inward_ratio,'descend');

        if sum(degreeOut) <= 3000
        else
            for j = 1 : length(ind)
                if sum(degreeOut(ind(1:j))) > 3000
                    newI = newI(ind(1:j));
                    break;
                end
            end
        end
        % one round BFS
        [~,newI] = BFS(graph,newI,1);
        I = union(I,newI,'stable');
        iter = iter + 1;
    end
    I2 = union(seeds,I,'stable');
end

if length(I2) > 5000
    p = zeros(1,length(I2));
    [~,ind] = intersect(I2,seeds);
    subgraph = graph(I2,I2);
    p(ind) = 1/length(ind);
    Prob = RandomWalk(subgraph,p,3);
    [~,ind2] = sort(Prob,'descend');
    Idx = ind2(1:5000);
    I2 = I2(Idx);
    I2 = union(seeds,I2,'stable');
end

end
