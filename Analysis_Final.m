N = 256;
filename = strcat('eigenvector3_', num2str(N));
filename = strcat(filename, '.txt');

x = load(filename);
x = flipud(x);
EigenVector = x(:,1:2:2*N) + 1i * x(:,2:2:2*N);
surf(abs(EigenVector))
axis square
axis([0 N 0 N])
set(gca,'FontSize',14,'FontWeight','bold','linewidth',1.5)
shading interp
colorbar
%caxis([-4/N 4/N])


np = [1, 4, 9, 16, 25, 36, 64];
Time480 = [378.4, 97.2, 44.8, 27.5, 18.5, 15.0, 15.3];
Time240 = [94.0, 24.5, 11.7, 7.9, 6.1, 6.3, 5.5];
Time120 = [23.5, 6.1, 3.1, 2.1, 1.7, 1.8, 2.6];
Time960 = [1505.0, 404.7, 184.1, 123.1, 89.3, 60.2, 59.4];
figure()
plot(np, Time480(1)./Time480, 'mo',... 
       'LineWidth',2,...
       'MarkerSize',10,...
       'MarkerEdgeColor','r',...
       'MarkerFaceColor',[1,0.8,0]);
hold on;

plot(np, np, '--','LineWidth',1.5)
set(gca,'FontSize',16,'FontWeight','bold','linewidth',1.5);
set(gca,'XTick',[1, 4, 9, 16, 25, 36, 64],'tickdir','in');
set(gca,'xtickLabel',{'1', '4', '9', '16', '25', '36', '64'});
xlabh = xlabel('np (Number of processes)');
ylabh = ylabel('T(1) / T(np)');
set(xlabh,'Position',get(xlabh,'Position') - [0 .0012 0])
set(ylabh,'Position',get(ylabh,'Position') - [1.5 0 0])


B = [0.0001, 0.001, 0.01, 0.1, 0.3];
E0 = [0.02448, 0.02448, 0.02480, 0.05132, 0.1500];
E1 = [0.06120, 0.06072, 0.05687, 0.05657, 0.1517];
E2 = [0.09793, 0.06168, 0.06648, 0.06786, 0.1563];
E3 = [0.12240, 0.09804, 0.09516, 0.08594, 0.1671];
figure()
hold on
plot(log(B)/log(10),E0);
plot(log(B)/log(10),E1);
plot(log(B)/log(10),E2);
plot(log(B)/log(10),E3);


