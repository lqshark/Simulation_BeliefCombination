function y = get_random_beliefs(num_of_dimensions, num_of_agents) % get random beliefs for agents

agent = cell(num_of_agents,1); % use Cell Matrix to store array with different length
for i = 1:num_of_agents
    
    num_of_states = randi([1,(2.^num_of_dimensions)]);
    all_states = get_all_states(num_of_dimensions);
    
    index_chosenState = sort(randperm(2.^num_of_dimensions,num_of_states));
    belief = [];
    for i_belief = 1:length(index_chosenState)
        belief = [belief,all_states(index_chosenState(i_belief),:)];
    end
    agent{i,:} = belief;
end

y = agent;