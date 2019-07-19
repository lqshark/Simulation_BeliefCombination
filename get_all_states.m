function y = get_all_states(num_of_dimensions)                                                                    
n = 2^num_of_dimensions;                                                                 
index = 0:n-1;                                                            
index = index';                                                       
nBin = dec2bin(index);                                                   
all_states = zeros(n,num_of_dimensions);
for i = 1:n
    temp = nBin(i,:)';
    tempRow = str2num(temp)';                                            
    all_states(i,:) = tempRow; 
end
y = all_states;