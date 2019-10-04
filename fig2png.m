imagePath = 'D:\software\GitHub\Simulation_integrated_version\d10newPlots\';  % ͼƬ���·��
imageFiles = dir([imagePath '*.fig']);
numFiles = length(imageFiles);
for i=1:numFiles                 % ��3��ʼ����Ϊǰ�����ǵ�ǰ·��j.1����һ��·����..��
    imageFile = strcat(imagePath,imageFiles(i).name);
    open(imageFile); 
    set(gcf,'position',[50,250,400,400]);
    pause(1);
    [filepath,name,ext] = fileparts(imageFiles(i).name);
    saveas(figure(1), ['D:\software\GitHub\Simulation_integrated_version\d10newPlots2png\',name,'.png']);
    close; 
    
end