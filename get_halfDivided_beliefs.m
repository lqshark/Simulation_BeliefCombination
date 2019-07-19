function y = get_halfDivided_beliefs(num_of_dimensions, num_of_agents, bias) % get halfDivided beliefs for agents

%% give propotional bias believes to 100 agents initially
% ***INPUT-VARIABLE*** bias of belief 1
agent = cell(num_of_agents,1); % use Cell Matrix to store array with different length
for i = 1 : num_of_agents * bias
    agent{i,:} = get_belief1(num_of_dimensions);
end
for i = (num_of_agents * bias + 1) : num_of_agents
    agent{i,:} = get_belief2(num_of_dimensions);
end

y = agent;