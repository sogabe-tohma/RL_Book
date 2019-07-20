function draw_a_grid(v, matrixRow, matrixCol,start,goal,barrier,N_goal)
     arr=zeros(matrixRow,matrixCol,4);
     for r=1:matrixRow
        for c=1:matrixCol 
             if v(r,c)~=0
                 a=v(r,c);
                 arr(r,c,a)=1;
             end
        end
     end
     ParseArrows(arr, matrixRow, matrixCol,start,goal,barrier,N_goal)
end
