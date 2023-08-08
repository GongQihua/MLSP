function [dataset] = loaddata(path)

for i = 1:24
    name = strcat(path, int2str(i),'.txt');
    data = load(name);
    dataset{i} = data';
    
    for j = 1:size(dataset{i},2)%normalizing
        dataset{i}(:,j) = dataset{i}(:,j)/norm(dataset{i}(:,j));
    
    end
end

end