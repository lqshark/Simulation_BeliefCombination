function y = get_totallyIgnorant_beliefs(num_of_dimensions, num_of_agents) % get totallyIgnorant beliefs for agents

agent = cell(num_of_agents,1); % use Cell Matrix to store array with different length
all_states = get_all_states(num_of_dimensions);
belief = [];
for i_connect = 1 : 2^num_of_dimensions
    belief = [belief,all_states(i_connect,:)];
end

for i = 1:num_of_agents
    agent{i,:} = belief;
end

y = agent;