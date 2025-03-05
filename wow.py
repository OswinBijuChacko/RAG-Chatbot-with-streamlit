r=3
c=3
m=[[0,1,0],[1,1,1],[1,1,1]]
parking=[]
for i in m:
    count=0
    for j in i:
        if j == 1:
            count+=1
    parking.append(count)
print(parking[parking.index(max(parking))])
print('total slots available')

        