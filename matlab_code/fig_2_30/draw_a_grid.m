function draw_a_grid(v, matrixRow, matrixCol,start,goal,barrier,N_goal)
     v(goal.row,goal.col)=0;
     v(barrier.row,barrier.col)=0;
     v(N_goal.row,N_goal.col)=0;
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
