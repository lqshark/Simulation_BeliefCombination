close all
clc
clear all
tic
%% input
num_of_agents = 100; % ***INPUT-VARIABLE*** total 100 agents in the system, each agent has
% one belief a1={};a2={};...;a100={}.

num_of_dimensions = 10;% ***INPUT-VARIABLE*** number of binary values in one state(x,y,z,...)

times_runs = 50; % ***INPUT-VARIABLE***
times_iteration = 2000; % ***INPUT-VARIABLE***

%% flag
flag_initial = 1; % 0: random; 1: totally ignorant; 2: singleton; 3: half-divided.

flag_pooling = 0; % 0: without pooling; 1: with pooling.
flag_poolingOperator = 0; % 0: max Similarity; 1: min Similarity; 2: average Similarity.

flag_updating = 1; % 0: without evidential updating; 1: with evidential updating.
flag_updatingOperator = 1; % 0: Delete operator; 1: Modify operator.

flag_dynamic = 0; % 0: static environment True state do not change; 1: dynamic environment.
%% true state
if flag_dynamic == 0
    T = ones(1,num_of_dimensions); % True state we assumed
end
%% display of Process operator (show in graphs)
if flag_initial == 0
    ope1 = "Random, ";
end
if flag_initial == 1
    ope1 = "Totally ignorant, ";
end
if flag_initial == 2
    ope1 = "Singleton, ";
end
if flag_initial == 3
    ope1 = "Half-divided, ";
end

if flag_pooling == 0
    ope2 = "";
else
    if flag_poolingOperator == 0
        ope2 = "maxSim, ";
    end
    if flag_poolingOperator == 1
        ope2 = "minSim, ";
    end
    if flag_poolingOperator == 2
        ope2 = "avrSim, ";
    end
end

if flag_updating == 0
   ope3 = "";
else
    if flag_updatingOperator == 0
        ope3 = "Delete, ";
    end
    if flag_updatingOperator == 1
        ope3 = "Modify, ";
    end
end

operator = ope1 + ope2 + ope3;

%% heatmap loop

r = 1 : -0.2 : 0; % ***INPUT-VARIABLE*** evidence rate
r(6) = 0.01;
epsilon = 0 : 0.1 : 0.5; % ***INPUT-VARIABLE*** noise rate
% 6 * 6 heatmap
heat_Hsim = zeros(6);
heat_times_avr = zeros(6);
flag_cons = zeros(6,6, times_runs);
times_convergence = zeros(6,6, times_runs);
times_consensus = zeros(6,6, times_runs);

for i_heatcol = 1 : 6
    for i_heatrow = 1 : 6
        
        % runs and iteration
        rate_conv = zeros(times_runs, times_iteration);
        flag_conv = zeros(times_runs, 1);
        result = cell(times_runs, 1);
        cardinality = zeros(times_runs,times_iteration);
        Hsim = zeros(times_runs,times_iteration);
        for i_runs = 1:times_runs
            
            %% initialization
            if flag_initial == 0
                agent = get_random_beliefs(num_of_dimensions, num_of_agents);
            end
            if flag_initial == 1
                agent = get_totallyIgnorant_beliefs(num_of_dimensions, num_of_agents);
            end
            if flag_initial == 2
                agent = get_singleton_beliefs(num_of_agents);
            end
            if flag_initial == 3
                bias = 0.6;
                agent = get_halfDivided_beliefs(num_of_dimensions, num_of_agents, bias);
            end
            
            % set a counter for counting the number of iteration
            counter = 0;
            flag_converge_inside = 1;
            
            cardinality_temp = zeros(1,num_of_agents);
            Hsim_temp = zeros(1,num_of_agents);
            
            while counter<times_iteration
                % in the beginning of every iterative, calculate the cardinality of
                % states.
                agent_last = agent; % use for convergence times counting
                for i_cardinality_combine = 1: num_of_agents
                    cardinality_temp(i_cardinality_combine) = ...
                        length(agent{i_cardinality_combine})/num_of_dimensions;
                end
                cardinality(i_runs, counter+1) = mean(cardinality_temp(:));
                
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
                Hsim(i_runs, counter+1) = mean(Hsim_temp(:));
                
                %%%%%%%%%%%%%%%%POOLING PROCESS%%%%%%%%%%%
                % choose 2 agents for info-combining randomly
                if flag_pooling == 1
                    
                    No_agent_pick = randperm(num_of_agents,2);
                    
                    agent_cal = {agent{No_agent_pick(1),:},agent{No_agent_pick(2),:}} ;
                    
                    % get all possible states for comparison
                    all_states = get_all_states(num_of_dimensions);
                    Similarity_temp = cell(1,2);
                    
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
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Pooling Operator%%%%%%%%%%%%%%%%%%%
                        if flag_poolingOperator == 0
                            Similarity_No1_agent_pick = max(Similarity_temp{1});
                            Similarity_No2_agent_pick = max(Similarity_temp{2});
                        end
                        if flag_poolingOperator == 1
                            Similarity_No1_agent_pick = min(Similarity_temp{1});
                            Similarity_No2_agent_pick = min(Similarity_temp{2});
                        end
                        if flag_poolingOperator == 2
                            Similarity_No1_agent_pick = mean(Similarity_temp{1});
                            Similarity_No2_agent_pick = mean(Similarity_temp{2});
                        end
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
                end
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                %%%%%%%%%%EVIDENTIAL-UPDATING PROCESS%%%%%%%%%%%
                if flag_updating == 1
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
                                evidence = randsrc(1, 1, [~T(col_choose), T(col_choose); epsilon(i_heatrow), (1 - epsilon(i_heatrow))]);
                                %%%%%%%%%%%%%%%%%%Updating Operator%%%%%%%%%%%%%%%%%%%%%%
                                % Method 1: delete inconsistent states
                                if flag_updatingOperator == 0
                                    id = agent_eliminate(:, col_choose) ~= evidence;
                                    agent_eliminate(id, :) = [];
                                end
                                % or Method 2: modify inconsistent states
                                if flag_updatingOperator == 1
                                    id = agent_eliminate(:, col_choose) ~= evidence;
                                    agent_eliminate(id, col_choose) = evidence;
                                    agent_eliminate = unique(agent_eliminate,'rows');
                                end
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
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                counter = counter + 1; % counter of while loop
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % verify convergence
            rate_convergence = 0;
            for i_verify2 = 1 : num_of_agents
                if isequal(agent{i_verify2}, agent_last{i_verify2})
                    rate_convergence = rate_convergence + 1;
                end
            end
            rate_convergence = rate_convergence / num_of_agents;
            rate_conv(i_runs, counter) = rate_convergence; % use for counting convergence times
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % verify consensus
            flag_consensus = 0;
            for i_verify = 1 : num_of_agents
                if ~isequal(agent{1}, agent{i_verify})
                    break;
                end
            end
            if i_verify == num_of_agents
                times_consensus(i_heatrow, i_heatcol, i_runs) = counter;
                flag_consensus = 1;
            end
            flag_cons(i_heatrow, i_heatcol, i_runs) = flag_consensus; % use for counting consensus times
            if flag_cons(i_heatrow, i_heatcol, i_runs) == 0
                result{i_runs,:} = [-1, -1]; % represent not consensus
            else
                result{i_runs,:} = agent{1,:};
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
            
            
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for i = 1 : times_runs
            find_tmp = find(rate_conv(i, :) < 1);
            max_find = max(find_tmp);
            if max_find == times_iteration
                times_convergence(i_heatrow, i_heatcol, i) = 999999;
            else
                times_convergence(i_heatrow, i_heatcol, i) = max_find + 1;
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        heat_Hsim(i_heatrow, i_heatcol) = mean(Hsim(:));
        
        heat_times_avr(i_heatrow, i_heatcol) = mean(times_convergence(i_heatrow, i_heatcol, :));
        if heat_times_avr(i_heatrow, i_heatcol) > times_iteration
            heat_times_avr(i_heatrow, i_heatcol) = -1;
        end
        
        
    end
end

figure(1);
set(gcf,'position',[50,250,500,500]);
heatmap(epsilon,r,heat_Hsim')
xlabel("Noise rate")
ylabel("Evidence rate")
title( { ['Heatmap of Avr Similarity(B, T)',],operator,['Number of agents = ', num2str(num_of_agents),...
    ', Dimension = ',num2str(num_of_dimensions)],...
    ['Iterations = ', num2str(times_iteration),...
    ', Runs = ', num2str(times_runs) ]})

figure(2);
set(gcf,'position',[550,250,500,500]);
heatmap(epsilon,r,heat_times_avr')
xlabel("Noise rate")
ylabel("Evidence rate")
title( { ['Heatmap of Avr Convergence Times',],operator,['Number of agents = ', num2str(num_of_agents),...
    ', Dimension = ',num2str(num_of_dimensions)],...
    ['Iterations = ', num2str(times_iteration),...
    ', Runs = ', num2str(times_runs) ]})

toc