function y = get_belief2(num_of_dimensions) % get a bias belief for an agent

all_states = get_all_states(num_of_dimensions);

belief2 = [];

for i = (2 ^ (num_of_dimensions - 1) + 1) : 2 ^ num_of_dimensions
    belief2 = [belief2,all_states(i,:)];
end
y = belief2;