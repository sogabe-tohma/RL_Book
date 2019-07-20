function action = random_action(a)
    p=rand(1);
    eps=0.1;
    if p<(1-eps)
        action=a;
    else
        action=randi([1,4],1);
    end
end

