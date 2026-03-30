function angle=anglerot2vecmat(a,b)
a=a';b=b';

c = zeros(max(size(a),size(b)));
c(1,:) = a(2,:).*b(3,:)-a(3,:).*b(2,:);
c(2,:) = a(3,:).*b(1,:)-a(1,:).*b(3,:);
c(3,:) = a(1,:).*b(2,:)-a(2,:).*b(1,:);

d = sum(a.*b,1);

angle=atan2d(norm(c),d);