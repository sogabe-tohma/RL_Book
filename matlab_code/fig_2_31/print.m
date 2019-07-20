function print(v,row,col)
     for r=1:row
         disp("---------------------------------")
         d=zeros(4);
        for c=1:col
            d(c)=v(r,c);
        end
        fprintf("%5d   %5d    %5d    %5d\n",round(d(1)),round(d(2)),round(d(3)),round(d(4)))
     end 
     disp("---------------------------------")   
end
