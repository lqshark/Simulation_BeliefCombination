close all
clc
clear all
tic
%% number of agents and dimensions
num_of_agents = 100; % ***INPUT-VARIABLE*** total 100 agents in the system, each agent has 
                     % one belief a1={};a2={};...;a100={}.

num_of_dimensions = 2;% ***INPUT-VARIABLE*** number of binary values in one state(x,y,z,...)

% num_of_states % number of states in one belief{(),(),(),...} given in
% get_random_beliefs.m
T = ones(1,num_of_dimensions); % True state we assumed
flag_pooling = 0;

%% heatmap loop
r = 1 : -0.2 : 0; % ***INPUT-VARIABLE*** evidence rate
% r(6) = 0.01;
epsilon = 0 : 0.1 : 0.5; % ***INPUT-VARIABLE*** noise rate
% 6 * 6 heatmap
heat_Hsim = zeros(6);
heat_convTimes = zeros(6);



%% iterative 
times_outside = 50; % ***INPUT-VARIABLE***
times_inside = 1500; % ***INPUT-VARIABLE*** e.g., d = 2 or 3, times_inside = 500;

flag_conv = zeros(times_outside, 1);
result = cell(times_outside, 1);
cardinality = zeros(times_outside,times_inside);
Hsim = zeros(times_outside,times_inside);
for i_outside = 1:times_outside
    %% give random believes to 100 agents initially
    agent = cell(num_of_agents,1); % use Cell Matrix to store array with different length
    for i = 1:num_of_agents
        agent{i,:} = get_random_beliefs(num_of_dimensions);
    end
    % set a counter for counting the number of iterative times
    counter = 0;
    flag_converge_inside = 1;
    
    cardinality_temp = zeros(1,num_of_agents);
    Hsim_temp = zeros(1,num_of_agents);

while counter<times_inside
    % in the beginning of every iterative, calculate the cardinality of
    % states.

    for i_cardinality_combine = 1: num_of_agents
        cardinality_temp(i_cardinality_combine) = ...
            length(agent{i_cardinality_combine})/num_of_dimensions;
    end
    cardinality(i_outside, counter+1) = mean(cardinality_temp(:));
    
    % in the beginning of every iterative, calculate the similarity of
    % states.
    for i_similarity_combine = 1: num_of_agents
        num_of_states_sim = length(agent{i_similarity_combine}) / num_of_dimensions;
        T_sim = [];
        for i_sim = 1 : num_of_states_sim
            T_sim = [T_sim, T];
        end
        Hdis = sum( abs(T_sim - agent{i_similarity_combine}) );
        Hsim_temp(i_similarity_combine) = 1 - Hdis / length(agent{i_similarity_combine});
    end
    Hsim(i_outside, counter+1) = mean(Hsim_temp(:));

    %%%%%%%%%POOLING PROCESS%%%%%%%%%%%
    % choose 2 agents for info-combining randomly
    if flag_pooling == 1
        operator = "Pooling";
    No_agent_pick = randperm(num_of_agents,2);
    
    agent_cal = {agent{No_agent_pick(1),:},agent{No_agent_pick(2),:}} ;

    % get all possible states for comparison
    all_states = get_all_states(num_of_dimensions);
    Similarity_temp = cell(1,2); %%%%%%%%%%%%%%%%%%%%%typo%%%%%%%%%%%%%%%%%%%

    for i_all_states = 1:2^num_of_dimensions % traverse all possible states
        for i_chosen = 1:2
            for i_agent_cal = 1:(length(agent_cal{i_chosen})/num_of_dimensions)
                % traverse all states in beliefs of chosen agents
                Ham_distance = sum( abs( all_states(i_all_states,:) - ...
                    agent_cal{i_chosen}( (num_of_dimensions*(i_agent_cal-1)+1):...
                    (num_of_dimensions*i_agent_cal)) ) ) ;
                
                Similarity_temp{i_chosen}(i_agent_cal) = 1- Ham_distance/num_of_dimensions;
                
            end

        end
            Similarity_No1_agent_pick = max(Similarity_temp{1});
            Similarity_No2_agent_pick = max(Similarity_temp{2});
            Similarity(i_all_states) = min(Similarity_No1_agent_pick,Similarity_No2_agent_pick);
    end
    maxSimilarity = max(Similarity);
    No_of_maxSim_states = find(Similarity == maxSimilarity);

    % !!!!!!!!!!!!!!!!!!!!!take all of the maxSim states for the new
    % belief of the 2 agents (we chose each iterative)
    new_belief = [];
    for i_newBelief = 1: length(No_of_maxSim_states)
    new_belief = [new_belief, all_states(No_of_maxSim_states(i_newBelief),:)];
    end
        
    
    agent{No_agent_pick(1),:} = new_belief;
    agent{No_agent_pick(2),:} = new_belief;
    else
        operator = "";
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%EVIDENTIAL-UPDATING PROCESS%%%%%%%%%%%
    for i_eliminate = 1 : num_of_agents
        % use "if" to choose the agent has uncertain beliefs
        if length(agent{i_eliminate}) > num_of_dimensions
            num_of_states = length(agent{i_eliminate}) / num_of_dimensions;
            
            row = 1;
            col = 1;
            col_investigate = [];
            agent_eliminate = [];
            for i_convert = 1 : length(agent{i_eliminate})
                agent_eliminate(row,col) = agent{i_eliminate}(i_convert);
                if mod(i_convert, num_of_dimensions) == 0
                    row = row + 1;
                    col = 1;
                else
                    col = col + 1;
                end
            end
            for i_col = 1 : num_of_dimensions
                if sum(agent_eliminate(:, i_col)) ~= 0 && ...
                        sum(agent_eliminate(:, i_col)) ~= num_of_states
                    % this position is uncertain, need to investigate, 
                    % record it
                    col_investigate = [col_investigate, i_col];
                end
            end
            index_rand = randperm(length(col_investigate));
            % get a random position to investigate from all uncertain positions of this agent
            col_choose = col_investigate(index_rand(1)); 
            % use randsrc to give evidence rate
            if randsrc(1, 1, [0, 1; (1 - r(i_heatcol)), r(i_heatcol)]) == 1 % then find the evidence and do updating
                % evidence could be
                evidence = randsrc(1, 1, [0, 1; epsilon(i_heatrow), (1 - epsilon(i_heatrow))]);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Method 1: delete inconsistent states
%                 id = agent_eliminate(:, col_choose) ~= evidence;
%                 agent_eliminate(id, :) = [];
                
                % or Method 2: modify inconsistent states
                id = agent_eliminate(:, col_choose) ~= evidence;
                agent_eliminate(id, col_choose) = evidence;
                agent_eliminate = unique(agent_eliminate,'rows');
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                agent_temp = [];
                for i_convertback = 1 : size(agent_eliminate, 1)
                    agent_temp = [agent_temp, agent_eliminate(i_convertback, :)];
                end
                % update the original agent cell matrix
                agent{i_eliminate} = agent_temp;
            end
            
            
            

        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    counter = counter + 1; % counter of while loop
    
end
    
% verify convergence
flag_converge = 1;
for i_verify = 1 : num_of_agents
    if ~isequal(agent{1}, agent{i_verify})
        flag_converge = 0;
        break;
    end
end
flag_conv(i_outside) = flag_converge; % use for counting convergence times
if flag_conv(i_outside) == 0
    result{i_outside,:} = [-1, -1]; % represent not converge
else
    result{i_outside,:} = agent{1,:};
end

end


heat_Hsim(i_heatrow, i_heatcol) = mean(Hsim(:));

% operator = operator + " + Delete";
operator = operator + " + Modify";
heatmap(epsilon,r,heat_Hsim)
xlabel("Noise rate")
ylabel("Evidence rate")
title( { ['Avr Similarity',', Initial: Random'],operator,['Number of agents = ', num2str(num_of_agents),...
    ', Dimension = ',num2str(num_of_dimensions)],...
    ['Inside iterative times = ', num2str(times_inside),...
    ', Outside iterative times = ', num2str(times_outside) ]})

%% figure 1: cardinality curves overlap
counter_plot = 0:(times_inside-1);
for i_plot = 1:times_outside
    figure(1);
    set(gcf,'position',[50,250,500,500]);
    % figure('position',[200,200,500,500]);
    plot(counter_plot,cardinality(i_plot,:))
    ylim([0 5]) % d = 2
    % ylim([0 10]) % d = 3
    % ylim([0 20]) % d = 4
    % ylim([0 35]) % d = 5
    % ylim([0 1030]) % d = 10
    xlabel('Number of iterative times');
    ylabel('Average cardinality');
    hold on
end
set(get(gca,'title'),'FontSize',10,'FontWeight','normal')
title( { ['Number of agents = ', num2str(num_of_agents),...
    ', Dimension = ',num2str(num_of_dimensions),...
    ', Evidence rate = ',num2str(r),...
    ', Noise rate = ',num2str(epsilon),],...
    ['Inside iterative times = ', num2str(times_inside),...
    ', Outside iterative times = ', num2str(times_outside) ]})


for i_EB = 1:times_inside
cardinality_avr(i_EB) = mean(cardinality(:,i_EB));
cardinality_var(i_EB) = var(cardinality(:,i_EB));
end
k = 1;
for i_sample = 1:50:times_inside
    cardinality_avr_EB(k) = cardinality_avr(i_sample);
    cardinality_var_EB(k) = cardinality_var(i_sample);
    counter_plot_EB(k) = counter_plot(i_sample);
    k = k+1;
end
cardinality_avr_EB(k) = cardinality_avr(times_inside);
cardinality_var_EB(k) = cardinality_var(times_inside);
counter_plot_EB(k) = counter_plot(times_inside);
%% figure 2: avr cardinality with errorbar
figure(2);
set(gcf,'position',[550,250,500,500]);
errorbar(counter_plot_EB, cardinality_avr_EB, cardinality_var_EB)
ylim([0 5]) % d = 2
% ylim([0 10]) % d = 3
% ylim([0 20]) % d = 4
% ylim([0 35]) % d = 5
% ylim([0 1030]) % d = 10

xlabel('Number of iterative times');
ylabel('Average cardinality');
set(get(gca,'title'),'FontSize',10,'FontWeight','normal')
title( { ['Number of agents = ', num2str(num_of_agents),...
    ', Dimension = ',num2str(num_of_dimensions),...
    ', Evidence rate = ',num2str(r),...
    ', Noise rate = ',num2str(epsilon),],...
    ['Inside iterative times = ', num2str(times_inside),...
    ', Outside iterative times = ', num2str(times_outside) ]})


%% figure 3: cardinality curves overlap
for i_plot2 = 1:times_outside
    figure(3);
    set(gcf,'position',[50,250,500,500]);
    % figure('position',[200,200,500,500]);
    plot(counter_plot,Hsim(i_plot2,:))
    ylim([0 2]) 

    xlabel('Number of iterative times');
    ylabel('Average similarity(B, T)');
    hold on
end
set(get(gca,'title'),'FontSize',10,'FontWeight','normal')
title( { ['Number of agents = ', num2str(num_of_agents),...
    ', Dimension = ',num2str(num_of_dimensions),...
    ', Evidence rate = ',num2str(r),...
    ', Noise rate = ',num2str(epsilon),],...
    ['Inside iterative times = ', num2str(times_inside),...
    ', Outside iterative times = ', num2str(times_outside) ]})


for i_EB2 = 1:times_inside
Hsim_avr(i_EB2) = mean(Hsim(:,i_EB2));
Hsim_var(i_EB2) = var(Hsim(:,i_EB2));
end
k = 1;
for i_sample2 = 1:50:times_inside
    Hsim_avr_EB(k) = Hsim_avr(i_sample2);
    Hsim_var_EB(k) = Hsim_var(i_sample2);
    counter_plot_EB(k) = counter_plot(i_sample2);
    k = k+1;
end
Hsim_avr_EB(k) = Hsim_avr(times_inside);
Hsim_var_EB(k) = Hsim_var(times_inside);
counter_plot_EB(k) = counter_plot(times_inside);

%% figure 4: avr cardinality with errorbar
figure(4);
set(gcf,'position',[550,250,500,500]);
errorbar(counter_plot_EB, Hsim_avr_EB, Hsim_var_EB)
ylim([0 2]) 

xlabel('Number of iterative times');
ylabel('Average similarity(B, T)');
set(get(gca,'title'),'FontSize',10,'FontWeight','normal')
title( { ['Number of agents = ', num2str(num_of_agents),...
    ', Dimension = ',num2str(num_of_dimensions),...
    ', Evidence rate = ',num2str(r),...
    ', Noise rate = ',num2str(epsilon),],...
    ['Inside iterative times = ', num2str(times_inside),...
    ', Outside iterative times = ', num2str(times_outside) ]})

%% figure 5: histogram of result frequency(final graph)
figure(5);
set(gcf,'position',[1050,250,500,500]);

% result analysis
uni_result{1} = result{1};
j = 1;
count_uni = ones(1, times_outside); 
for i = 2 : times_outside
    flag = 0;
    for i_j = 1 : j
        if (result{i} == uni_result{i_j})
            count_uni(i_j) = count_uni(i_j) + 1;
            flag = 1;
            break;
        end
    end
    if (flag == 0)
        j = j + 1;
        uni_result{j} = result{i};
    end
end
label = cell(length(uni_result), 1);
b=bar(count_uni(1: j));
grid on;
for i_label = 1 : length(uni_result)
    if uni_result{i_label} == [-1, -1]
        label{i_label} = "[notConv]"; % represent not converging
                                      % in given combining times.
    else
        label{i_label} = "["+ num2str(uni_result{i_label})+"]";
    end
end
xticklabels(label);
xlabel('possible results');
ylabel('frequency');
set(get(gca,'title'),'FontSize',10,'FontWeight','normal')
title( { ['Number of agents = ', num2str(num_of_agents),...
    ', Dimension = ',num2str(num_of_dimensions),...
    ', Evidence rate = ',num2str(r),...
    ', Noise rate = ',num2str(epsilon),],...
    ['Inside iterative times = ', num2str(times_inside),...
    ', Outside iterative times = ', num2str(times_outside) ]})


toc