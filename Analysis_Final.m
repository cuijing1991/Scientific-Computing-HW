N = 128;
filename = strcat('eigenvector0_', num2str(N));
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