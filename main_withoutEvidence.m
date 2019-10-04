close all
clc
clear all
tic
%% input
num_of_agents = 100; % ***INPUT-VARIABLE*** total 100 agents in the system, each agent has
% one belief a1={};a2={};...;a100={}.

num_of_dimensions = 6;% ***INPUT-VARIABLE*** number of binary values in one state(x,y,z,...)

times_runs = 100; % ***INPUT-VARIABLE***
times_iteration = 2000; % ***INPUT-VARIABLE***

r = 0.1; % ***INPUT-VARIABLE*** evidence rate
epsilon = 0.1; % ***INPUT-VARIABLE*** noise rate
%% flag
flag_initial = 3; % 0: random; 1: totally ignorant; 2: singleton; 3: half-divided.

flag_pooling = 1; % 0: without pooling; 1: with pooling.
flag_poolingOperator = 0; % 0: max Similarity; 1: min Similarity; 2: average Similarity.

flag_updating = 0; % 0: without evidential updating; 1: with evidential updating.
flag_updatingOperator = 0; % 0: Delete operator; 1: Modify operator.

flag_dynamic = 0; % 0: static environment True state do not change; 1: dynamic environment.
%% true state
if flag_dynamic == 0
    T = ones(1,num_of_dimensions); % True state we assumed
end
%% display of Process operator
if flag_initial == 0
    ope1 = "Random, ";
end
if flag_initial == 1
    ope1 = "Totally ignorant, ";
end
if flag_initial == 2
    ope1 = "Singleton(0.4,0.2,0.2,0.2), ";
end
if flag_initial == 3
    ope1 = "Half-divided(bias = 0.6), ";
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

notCons = [];
for i = 1 : num_of_dimensions
    notCons = [notCons, -1];
end
flag_cons = zeros(times_runs);
times_convergence = zeros(times_runs);
times_consensus = zeros(times_runs);

    % runs and iteration
    rate_conv = zeros(times_runs, times_iteration);

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
                            if randsrc(1, 1, [0, 1; (1 - r), r]) == 1 % then find the evidence and do updating
                                % evidence could be
                                evidence = randsrc(1, 1, [~T(col_choose), T(col_choose); epsilon, (1 - epsilon)]);
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
                times_consensus(i_runs) = counter;
                flag_consensus = 1;
            end
            flag_cons(i_runs) = flag_consensus; % use for counting consensus times
            if flag_cons(i_runs) == 0
                result{i_runs,:} = notCons; % represent not consensus
            else
                result{i_runs,:} = agent{1,:};
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    percentile = 0.1;
    up = int32((1 - percentile/2) * times_runs);
    low = int32(percentile/2 * times_runs);
    if up < 1
        up = 1;
    end
    if up > times_runs
        up = times_runs;
    end
    if low < 1
        low = 1;
    end
    if low > times_runs
        low = times_runs;
    end
    
    for i_EB2 = 1:times_iteration
        tmp = sort(Hsim(:,i_EB2));
        Hsim_upbound(i_EB2) = tmp(up);
        Hsim_lowbound(i_EB2) = tmp(low);
        Hsim_avr(i_EB2) = mean(Hsim(:,i_EB2));
        
        tmp2 = sort(cardinality(:,i_EB2));
        cardinality_upbound(i_EB2) = tmp2(up);
        cardinality_lowbound(i_EB2) = tmp2(low);
        cardinality_avr(i_EB2) = mean(cardinality(:,i_EB2));
    end
    
    counter_plot = 0:(times_iteration-1);
    k = 1;
    for i_sample2 = 1:100:times_iteration
        Hsim_avr_EB(k) = Hsim_avr(i_sample2);
        Hsim_upbound_EB(k) = Hsim_upbound(i_sample2) - Hsim_avr(i_sample2);
        Hsim_lowbound_EB(k) = Hsim_lowbound(i_sample2) - Hsim_avr(i_sample2);
        
        cardinality_avr_EB(k) = cardinality_avr(i_sample2);
        cardinality_upbound_EB(k) = cardinality_upbound(i_sample2) - cardinality_avr(i_sample2);
        cardinality_lowbound_EB(k) = cardinality_lowbound(i_sample2) - cardinality_avr(i_sample2);
        
        counter_plot_EB(k) = counter_plot(i_sample2);
        k = k+1;
    end
    Hsim_avr_EB(k) = Hsim_avr(times_iteration);
    Hsim_upbound_EB(k) = Hsim_upbound(times_iteration) - Hsim_avr(times_iteration);
    Hsim_lowbound_EB(k) = Hsim_lowbound(times_iteration) - Hsim_avr(times_iteration);
    
    cardinality_avr_EB(k) = cardinality_avr(times_iteration);
    cardinality_upbound_EB(k) = cardinality_upbound(times_iteration) - cardinality_avr(times_iteration);
    cardinality_lowbound_EB(k) = cardinality_lowbound(times_iteration) - cardinality_avr(times_iteration);
    counter_plot_EB(k) = counter_plot(times_iteration);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for i = 1 : times_runs
        find_tmp = find(rate_conv(i, :) < 1);
        max_find = max(find_tmp);
        if max_find == times_iteration
            times_convergence(i) = 999999;
        else
            times_convergence(i) = max_find + 1;
        end
    end


    times_avr= mean(times_convergence(:));
    if times_avr > times_iteration
        times_avr = -1;
    end
    
%% figure 1: cardinality curves overlap
for i_plot = 1:times_runs
    figure(1);
    set(gcf,'position',[50,250,500,500]);
    % figure('position',[200,200,500,500]);
    plot(counter_plot,cardinality(i_plot,:))
    % ylim([0 5]) % d = 2
    % ylim([0 10]) % d = 3
    % ylim([0 20]) % d = 4
    ylim([0 35]) % d = 5
    % ylim([0 1030]) % d = 10
    grid on
    xlabel('Number of iterations');
    ylabel('Average cardinality');
    hold on
end
set(get(gca,'title'),'FontSize',10,'FontWeight','normal')
title( {operator,['Number of agents = ', num2str(num_of_agents),...
    ', Dimension = ',num2str(num_of_dimensions)],...
    ['Iterations = ', num2str(times_iteration),...
    ', Runs = ', num2str(times_runs) ]})

%% figure 2: avr cardi with errorbar
figure(2);
set(gcf,'position',[550,250,500,500]);
errorbar(counter_plot_EB, cardinality_avr_EB, cardinality_lowbound_EB, cardinality_upbound_EB)

grid on

    % ylim([0 5]) % d = 2
    % ylim([0 10]) % d = 3
    % ylim([0 20]) % d = 4
    ylim([0 35]) % d = 5
    % ylim([0 1030]) % d = 10
xlabel('Number of iterations');
ylabel('Average cardinality');
set(get(gca,'title'),'FontSize',10,'FontWeight','normal')
title( {operator,['Number of agents = ', num2str(num_of_agents),...
    ', Dimension = ',num2str(num_of_dimensions)],...
    ['Iterations = ', num2str(times_iteration),...
    ', Runs = ', num2str(times_runs) ]})

%% figure 3: similarity curves overlap
for i_plot2 = 1:times_runs
    figure(3);
    set(gcf,'position',[50,250,500,500]);
    % figure('position',[200,200,500,500]);
    plot(counter_plot,Hsim(i_plot2,:))
    ylim([0 1]) 

    grid on
    xlabel('Number of iterations');
    ylabel('Average similarity(B, T)');
    hold on
end
set(get(gca,'title'),'FontSize',10,'FontWeight','normal')
title( {operator,['Number of agents = ', num2str(num_of_agents),...
    ', Dimension = ',num2str(num_of_dimensions)],...
    ['Iterations = ', num2str(times_iteration),...
    ', Runs = ', num2str(times_runs) ]})

%% figure 4: avr sim with errorbar
figure(4);
set(gcf,'position',[550,250,500,500]);
errorbar(counter_plot_EB, Hsim_avr_EB, Hsim_lowbound_EB, Hsim_upbound_EB)

grid on
ylim([0 1])
xlabel('Number of iterations');
ylabel('Average similarity(B, T)');
set(get(gca,'title'),'FontSize',10,'FontWeight','normal')
title( {operator,['Number of agents = ', num2str(num_of_agents),...
    ', Dimension = ',num2str(num_of_dimensions)],...
    ['Iterations = ', num2str(times_iteration),...
    ', Runs = ', num2str(times_runs) ]})

%% figure 5: histogram of result frequency(final graph)
figure(5);
set(gcf,'position',[1050,250,500,500]);

% result analysis
uni_result{1} = result{1};
j = 1;
count_uni = ones(1, times_runs); 
for i = 2 : times_runs
    flag = 0;
    for i_j = 1 : j
        if (isequal(result{i}, uni_result{i_j}))
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
    if uni_result{i_label}(1) == -1
        label{i_label} = "[notCons]"; % represent not consensus
    else
        label{i_label} = "["+ num2str(uni_result{i_label})+"]";
    end
end


set(gca,'XTickLabelRotation',0)
xticklabels(label);

xlabel('possible results');
ylabel('frequency');
set(get(gca,'title'),'FontSize',10,'FontWeight','normal')
title( {operator,['Number of agents = ', num2str(num_of_agents),...
    ', Dimension = ',num2str(num_of_dimensions)],...
    ['Iterations = ', num2str(times_iteration),...
    ', Runs = ', num2str(times_runs) ]})

toc