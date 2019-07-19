function y = get_belief1(num_of_dimensions) % get a bias belief for an agent

all_states = get_all_states(num_of_dimensions);

belief1 = [];

for i = 1 : 2 ^ (num_of_dimensions - 1)
    belief1 = [belief1,all_states(i,:)];
end

y = belief1;