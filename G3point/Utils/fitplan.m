function [A,B,C,distsigned,distabs]=fitplan(cloud)
% fit un plan de coordonnees z=ax+by+c
% par methode de decomposition en valeurs singuliere (fonction svd de
% matlab) Script issu de la fonction plane_fit de Kevin Moerman dans Matlab Central
% Z=(A*X)+(B*Y)+C; 
P=[mean(cloud(:,1)),mean(cloud(:,2)),mean(cloud(:,3))];
[U,S,V]=svd([cloud(:,1)-P(1),cloud(:,2)-P(2),cloud(:,3)-P(3)],0);
N=-1/V(end,end)*V(:,end);
A=N(1); B=N(2); C=-P*N;
distsigned=(C+A*cloud(:,1)+B*cloud(:,2)-cloud(:,3))/sqrt(A^2+B^2+1);
distabs=abs(C+A*cloud(:,1)+B*cloud(:,2)-cloud(:,3))/sqrt(A^2+B^2+1);