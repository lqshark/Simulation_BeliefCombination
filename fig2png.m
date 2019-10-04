imagePath = 'D:\software\GitHub\Simulation_integrated_version\d10newPlots\';  % 图片存放路径
imageFiles = dir([imagePath '*.fig']);
numFiles = length(imageFiles);
for i=1:numFiles                 % 从3开始，因为前两个是当前路径j.1和上一级路径‘..’
    imageFile = strcat(imagePath,imageFiles(i).name);
    open(imageFile); 
    set(gcf,'position',[50,250,400,400]);
    pause(1);
    [filepath,name,ext] = fileparts(imageFiles(i).name);
    saveas(figure(1), ['D:\software\GitHub\Simulation_integrated_version\d10newPlots2png\',name,'.png']);
    close; 
    
end