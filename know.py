p=[6,2]
quad1=min(p[0]-1,p[1]-1)
quad2=min(p[0]-1,8-p[1])
quad3=min(8-p[0],p[1]-1)
quad4=min(8-p[0],8-p[1])
print(quad1+quad2+quad3+quad4)