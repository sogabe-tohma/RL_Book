function y =soft_max(x,d_x)
    sum1=0;
    max1=0;
    if max1<max(d_x)
        max1=max(d_x);
    end
    disp(max1)
    exp1=exp(x-max1);
    for t_=1:length(d_x)
        sum1=sum1+exp(d_x(t_)-max1);
    end
    disp(sum1)
    y=exp1/sum1;
end