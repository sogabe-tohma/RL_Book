function y =soft_max(x)
    sum1=0;
    max1=0;
    y=[];
    if max1<max(x)
        max1=max(x);
    end 
    for t_=1:length(x)
        sum1=sum1+exp(x(t_)-max1);
    end
    for t=1:length(x) 
        exp1=exp(x(t)-max1);
        y=[y,exp1/sum1];
    end
end