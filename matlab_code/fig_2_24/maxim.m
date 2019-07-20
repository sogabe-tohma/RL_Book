function [val,place] = maxim(a)
    b=a;
    val=max(a(a~=0));
    for l=1:length(b)
        if b(l)==val
            place=l;
            break;
        else
            disp('')
        end
    end
end