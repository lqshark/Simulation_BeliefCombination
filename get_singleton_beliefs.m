function y = get_singleton_beliefs(num_of_agents) % get singleton beliefs for agents
num_of_dimensions = 2;
%% give uniform singleton bias believes to 100 agents initially
% x = 25; y = 50; z = 75; % bias = 0.25,0.25,0.25,0.25
% x = 40; y = 80; z = 90; % bias = 0.4,0.4,0.1,0.1
% x = 40; y = 70; z = 90; % bias = 0.4,0.3,0.2,0.1
x = 40; y = 60; z = 80; % bias = 0.4,0.2,0.2,0.2
% x = 70; y = 80; z = 90; % bias = 0.7,0.1,0.1,0.1
all_states = get_all_states(num_of_dimensions);
agent = cell(num_of_agents,1); % use Cell Matrix to store array with different length
for i = 1 : x * num_of_agents/100
    agent{i, :} = all_states(1,:);
end
for i = (x * num_of_agents/100 + 1) : y * num_of_agents/100
    agent{i, :} = all_states(2,:);
end
for i = (y * num_of_agents/100 + 1) : z * num_of_agents/100
    agent{i, :} = all_states(3,:);
end
for i = (z * num_of_agents/100 + 1) : num_of_agents
    agent{i, :} = all_states(4,:);
end

y = agent;